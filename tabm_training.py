"""Адаптированный пример использование tabm из коробки"""

import math
import warnings
from typing import Literal, NamedTuple

import numpy as np
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm

warnings.simplefilter('ignore')
from utils.tabm_reference import Model, make_parameter_groups
from config import N_EPOCHS, SEED

warnings.resetwarnings()

torch.manual_seed(SEED)


def train_tabm(x_train: np.ndarray, y_train: np.array, x_val: np.ndarray, y_val: np.array, fold: int) -> tuple:
    """
    Обучает модель tabm. Основан на примере обучения от авторов модели.

    :param x_train: Тренировочная выборка.
    :param y_train: Ground truth значения для трейна.
    :param x_val: Валидационная выборка.
    :param y_val: Ground truth значения для валидации.
    :param fold:
    :return: Модель и таргет трансформер.
    """
    TaskType = Literal['regression', 'binclass', 'multiclass']

    task_type: TaskType = 'regression'
    n_classes = None

    X_cont_train: np.ndarray = x_train
    Y_train: np.ndarray = y_train

    X_cont_val: np.ndarray = x_val
    Y_val: np.ndarray = y_val

    task_is_regression = task_type == 'regression'

    X_cont_train = X_cont_train.astype(np.float32)
    X_cont_val = X_cont_val.astype(np.float32)
    n_cont_features = X_cont_train.shape[1]

    X_cat = None

    Y_train = Y_train.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    data_numpy = {
        'train': {'x_cont': X_cont_train, 'y': Y_train},
        'val': {'x_cont': X_cont_val, 'y': Y_val},
    }
    cat_cardinalities = []

    class RegressionLabelStats(NamedTuple):
        mean: float
        std: float

    Y_train = data_numpy['train']['y'].copy()
    if task_type == 'regression':
        # For regression tasks, it is highly recommended to standardize the training labels.
        regression_label_stats = RegressionLabelStats(
            Y_train.mean().item(), Y_train.std().item()
        )
        Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
    else:
        regression_label_stats = None


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Convert data to tensors
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train, device=device)
    if task_type == 'regression':
        for part in data:
            data[part]['y'] = data[part]['y'].float()
        Y_train = Y_train.float()

    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True will result in faster training on compatible hardware.
    amp_enabled = False and amp_dtype is not None
    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

    # torch.compile
    compile_model = False

    # fmt: off
    print(
        f'Device:        {device.type.upper()}'
        f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
        f'\ntorch.compile: {compile_model}'
    )

    arch_type = 'tabm-mini'
    bins = rtdl_num_embeddings.compute_bins(data['train']['x_cont'])

    model = Model(
        n_num_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        backbone={
            'type': 'MLP',
            'n_blocks': 3 if bins is None else 2,
            'd_block': 512,
            'dropout': 0.1,
        },
        bins=bins,
        num_embeddings=(
            None
            if bins is None
            else {
                'type': 'PiecewiseLinearEmbeddings',
                'd_embedding': 16,
                'activation': False,
                'version': 'B',
            }
        ),
        arch_type=arch_type,
        k=32,
    ).to(device)
    optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)

    if compile_model:
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument
        # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
    def apply_model(part: str, idx: Tensor) -> Tensor:
        return (
            model(
                data[part]['x_cont'][idx],
                data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
            )
            .squeeze(-1)  # Remove the last dimension for regression tasks.
            .float()
        )

    base_loss_fn = F.mse_loss if task_type == 'regression' else F.cross_entropy

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        # TabM produces k predictions per object. Each of them must be trained separately.
        # (regression)     y_pred.shape == (batch_size, k)
        # (classification) y_pred.shape == (batch_size, k, n_classes)
        k = y_pred.shape[-1 if task_type == 'regression' else -2]
        return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))


    @evaluation_mode()
    def evaluate(part: str) -> float:
        model.eval()

        # When using torch.compile, you may need to reduce the evaluation batch size.
        eval_batch_size = 8096
        y_pred: np.ndarray = (
            torch.cat(
                [
                    apply_model(part, idx)
                    for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                ]
            )
            .cpu()
            .numpy()
        )

        if task_type == 'regression':
            # Transform the predictions back to the original label space.
            assert regression_label_stats is not None
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean
        # Compute the mean of the k predictions.
        if task_type != 'regression':
            # For classification, the mean must be computed in the probabily space.
            y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred = y_pred.mean(1)
        y_true = data[part]['y'].cpu().numpy()
        score = (
            -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
            if task_type == 'regression'
            else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
        )
        return float(score)  # The higher -- the better.

    print(f'Test score before training: {evaluate("val"):.4f}')

    n_epochs = N_EPOCHS
    patience = 16

    batch_size = 256
    epoch_size = math.ceil(len(y_train) / batch_size)
    best = {
        'val': -math.inf,
        'epoch': -1,
    }
    best_checkpoint = ''
    best_model = None
    # Early stopping: the training stops when
    # there are more than `patience` consequtive bad updates.
    patience = 10
    remaining_patience = patience

    print('-' * 88 + '\n')
    for epoch in range(n_epochs):
        for batch_idx in tqdm(
            torch.randperm(len(data['train']['y']), device=device).split(batch_size),
            desc=f'Epoch {epoch}',
            total=epoch_size,
        ):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()

        val_score = evaluate('val')
        print(f'(val) {val_score:.4f}')

        if val_score - best['val'] > 0.1:
            print('🌸 New best epoch! 🌸')
            best_checkpoint = f'checkpoints/best_model_{fold}'
            best_model = model
            torch.save(model.state_dict(), best_checkpoint)
            best = {'val': val_score, 'epoch': epoch}
            remaining_patience = patience

        else:
            remaining_patience -= 1

        if remaining_patience < 0:
            break

        print()

    print('\n\nResult:')
    print(best)
    return best_model, regression_label_stats
