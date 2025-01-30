"""Класс с солвером задачи"""

from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import SEED, ALGO, OPTUNA, OPTUNA_SCORER, N_THREADS, N_FOLDS, TIMEOUT, LEARN_LINEAR, N_TRIALS, \
    ENSEMBLE_MODELS
from tabm_training import train_tabm
from utils.dataset import Dataset


def calc_accuracy_score(y_true: np.array, preds: np.array, mid_prices: np.array) -> float:
    """
    Вычисляет accuracy.

    :param y_true: Ground truth значения.
    :param preds: Предикты модели
    :param mid_prices: Мид прайсы для перехода к исходным значениям
    :return: значение accuracy.
    """
    normalized_preds = preds + mid_prices
    normalized_preds = [round(pred / 5) * 5 for pred in normalized_preds]
    normalized_y_true = y_true + mid_prices
    accuracy = accuracy_score(normalized_y_true, normalized_preds)
    return accuracy


def get_stats(preds: np.array, y_true: np.array, mid_prices: np.array, task: str):
    """
    Вычисляет значение RMSE и accuracy для предиктов модели.

    :param preds: Предикты модели.
    :param y_true: Ground truth значения.
    :param mid_prices:
    :param task: Мид прайсы для перехода к исходным значениям
    :return:
    """
    train_loss = mean_squared_error(y_true, preds)
    train_accuracy = calc_accuracy_score(y_true, preds, mid_prices)
    print(f'{task} loss: {np.sqrt(train_loss)}, {task} accuracy: {train_accuracy}')


def custom_loss(y_true: np.array, y_pred: np.array, limit: int = 17, coef: int = 6) -> tuple:
    """

    :param y_true: Ground truth значения.
    :param y_pred: Предикты модели.
    :param limit: Граница сильного штрафа.
    :param coef: Коэффицент сильного штрафа
    :return: Градиент и гессиан.
    """
    diff = y_pred - y_true
    abs_diff = np.abs(diff)

    grad = np.zeros_like(diff)
    grad[abs_diff <= limit] = coef * diff[abs_diff <= limit]
    grad[abs_diff <= 2.5] = 0
    grad[abs_diff > limit] = np.sign(diff[abs_diff > limit])

    hess = np.zeros_like(diff)
    hess[abs_diff <= limit] = coef
    hess[abs_diff <= 2.5] = 0
    hess[abs_diff > limit] = 0

    return grad, hess


class Solver:
    """Солвер задачи"""

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset):
        self.train_dataset = train_dataset
        self.train_dataset.build_baseline_features()
        self.test_dataset = test_dataset
        self.seed = SEED

    def baseline_train(self,
                       x_train: np.ndarray,
                       y_train: np.array,
                       x_val: np.ndarray,
                       y_val: np.array,
                       mid_train: np.array,
                       mid_val: np.array,
                       algo: str = ALGO,
                       optuna: str = OPTUNA) -> tuple:
        """
        Запускает базовый алгоритм обучения разных моделей из коробки

        :param x_train: Тренировочная выборка.
        :param y_train: Ground truth значения для трейна.
        :param x_val: Валидационная выборка.
        :param y_val: Ground truth значения для валидации.
        :param mid_train: Мид прайсы для трейна.
        :param mid_val: Мид прайсы для валидации.
        :param algo: Тип используемого алгоритма.
        :param optuna: Используется ли оптюна.
        :return: Модель, скейлер для фичей, таргет трансформер.
        """
        print(f'Размерность трейна: {x_train.shape}')
        scaler = StandardScaler()  # MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        if algo == 'lin_reg':
            print('Выбран алгоритм линейной регрессии')
            model = Ridge()
        elif algo == 'random_forest':
            print('Выбран алгоритм случайного леса')
            model = RandomForestRegressor()
        elif algo == 'xgb':
            print('Выбран алгоритм xgboost')
            if optuna:
                print('Выбрано использование optuna')
                params = self.get_optuna_search(x_train, y_train, x_val, y_val, mid_train, mid_val, algo)
                params['objective'] = lambda y_true, y_pred: custom_loss(y_true, y_pred,
                                                                         params['limit'],  params['coef'])
                model = XGBRegressor(**params)
            else:
                model = XGBRegressor()
        elif algo == 'cat':
            print('Выбран алгоритм catboost')
            if optuna:
                print('Выбрано использование optuna')
                model = CatBoostRegressor(**self.get_optuna_search(x_train,
                                                                   y_train,
                                                                   x_val,
                                                                   y_val,
                                                                   mid_train,
                                                                   mid_val,
                                                                   algo))
            else:
                model = CatBoostRegressor()
        elif algo == 'lgbm':
            print('Выбран алгоритм lgbm')
            if optuna:
                print('Выбрано использование optuna')
                params = self.get_optuna_search(x_train, y_train, x_val, y_val, mid_train, mid_val, algo)
                params['objective'] = lambda y_true, y_pred: custom_loss(y_true, y_pred,
                                                                         params['limit'], params['coef'])
                model = LGBMRegressor(**params)
            else:
                model = LGBMRegressor()
        elif algo == 'knn':
            print('Выбран алгоритм knn')
            model = KNeighborsRegressor()
        elif algo == 'automl':
            print('Выбран алгоритм lightautoml')
            model = self.get_automl_training(x_train, y_train, x_val, y_val, mid_train, mid_val)
            return model, scaler, None
        elif algo == 'tabm':
            print('Выбран алгоритм tabm')
            model, target_transformer = train_tabm(x_train, y_train, x_val, y_val, 0)
            preds = self.get_predict_tabm(model, scaler, target_transformer, self.train_dataset.features_test)
            get_stats(preds, self.train_dataset.targets_test, self.train_dataset.mid_prices_test, 'Test')
            return model, scaler, target_transformer
        else:
            model = None
            raise Exception
        print('Старт обучения модели')
        model.fit(x_train, y_train)
        print('Окончание обучения модели')

        get_stats(model.predict(x_train), y_train, mid_train, 'Train')
        get_stats(model.predict(x_val), y_val, mid_val, 'Val')

        return model, scaler, None

    @staticmethod
    def get_optuna_search(x_train: np.ndarray,
                          y_train: np.array,
                          x_val: np.ndarray,
                          y_val: np.array,
                          train_mid_prices: np.array,
                          val_mid_prices: np.array,
                          model_type: str,
                          scorer: str = OPTUNA_SCORER):
        """
        Осуществляет поиск оптимальных гиперпараметров с помощью optuna.

        :param x_train: Тренировочная выборка.
        :param y_train: Ground truth значения для трейна.
        :param x_val: Валидационная выборка.
        :param y_val: Ground truth значения для валидации.
        :param train_mid_prices: Мид прайсы для трейна.
        :param val_mid_prices: Мид прайсы для валидации.
        :param model_type: Тип используемой модели.
        :param scorer: Оптимизируемая метрика.
        :return: Наилучшие гиперпараметры.
        """
        print('Запущен подбор гиперпараметров с optuna')
        direction = 'maximize' if scorer == 'accuracy' else 'minimize'
        study = optuna.create_study(study_name="study", direction=direction)

        def objective(trial):
            if model_type == 'xgb':
                param = {
                    'verbosity': 0,
                    'objective': lambda y_true, y_pred: custom_loss(y_true,
                                                                    y_pred,
                                                                    trial.suggest_int('limit', 2.5, 50),
                                                                    trial.suggest_int('coef', 1, 100)),
                    'booster': 'gbtree',
                    'eta': trial.suggest_float('eta', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),  # trial.suggest_int('max_depth', 3, 20)
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 10),
                    'lambda': trial.suggest_float('lambda', 1, 10),
                    'alpha': trial.suggest_float('alpha', 0, 10),
                }
                model = XGBRegressor(**param)
            elif model_type == 'cat':
                param = {
                    "verbose": False,
                    "iterations": trial.suggest_int("iterations", 200, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
                    "depth": trial.suggest_int("depth", 3, 15),
                    "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                }
                model = CatBoostRegressor(**param, random_state=SEED)
            elif model_type == 'lgbm':
                param = {
                    'objective': lambda y_true, y_pred: custom_loss(y_true,
                                                                    y_pred,
                                                                    trial.suggest_int('limit', 2.5, 50),
                                                                    trial.suggest_int('coef', 1, 100)),
                    'random_state': SEED,
                    "verbosity": -1,
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                    'colsample_bytree': trial.suggest_categorical('colsample_bytree',
                                                                  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                    'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
                    'learning_rate': trial.suggest_categorical('learning_rate',
                                                               [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, 100]),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                    'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100)
                }
                model = LGBMRegressor(**param)
            else:
                print('Ошибка с выбором модели')
                raise Exception

            model.fit(x_train, y_train)

            y_pred = model.predict(x_val)
            if scorer == 'accuracy':
                loss = calc_accuracy_score(y_val, y_pred, val_mid_prices)
                train_loss = calc_accuracy_score(y_train, model.predict(x_train), train_mid_prices)
                print(f'Train accuracy: {train_loss}, val accuracy: {loss}')
            else:
                loss = np.sqrt(mean_squared_error(y_val, y_pred))
                train_loss = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))
                print(f'Train rmse: {train_loss}, val_rmse: {loss}')
            return loss

        study.optimize(objective, n_trials=N_TRIALS)
        print(f'Окончен подбор гиперпараметров: {study.best_params}')
        return study.best_params

    @staticmethod
    def get_automl_training(x_train: np.ndarray,
                            y_train: np.array,
                            x_val: np.ndarray,
                            y_val: np.array,
                            mid_train: np.array,
                            mid_val: np.array) -> TabularAutoML:
        """
        Обучение lightautoml.

        :param x_train: Тренировочная выборка.
        :param y_train: Ground truth значения для трейна.
        :param x_val: Валидационная выборка.
        :param y_val: Ground truth значения для валидации.
        :param mid_train: Мид прайсы для трейна.
        :param mid_val: Мид прайсы для валидации.
        :return: Обученная модель.
        """
        train = pd.DataFrame(x_train)
        train['target'] = y_train
        train.columns = train.columns.astype(str)
        val = pd.DataFrame(x_val)
        val['target'] = y_val
        val.columns = val.columns.astype(str)
        task = Task('reg')
        roles = {
            'target': 'target',
        }
        automl = TabularAutoML(
            task=task,
            timeout=TIMEOUT,
            cpu_limit=N_THREADS,
            reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': SEED},
        )
        automl.fit_predict(train, roles=roles, verbose=1)
        get_stats(automl.predict(train).data[:, 0], y_train, mid_train, 'Train')
        get_stats(automl.predict(val).data[:, 0], y_val, mid_val, 'Val')
        return automl

    def predict_on_hold_out(self, model, scaler):
        """Замеряем итоговое качество алгоритма на отложенной выборке."""
        x_test = scaler.transform(self.train_dataset.features_test)
        preds = model.predict(x_test)
        try:
            get_stats(preds, self.train_dataset.targets_test, self.train_dataset.mid_prices_test, 'Test')
        except:
            get_stats(preds.data[:, 0], self.train_dataset.targets_test, self.train_dataset.mid_prices_test, 'Test')

    @staticmethod
    def get_predict_tabm(model: Any, scaler: Any, target_transformer: Any, x: np.ndarray):
        """
        Рассчитывает предикты моделью Tabm (tabm-mini).

        :param model: модель tabm.
        :param scaler: Скейлер.
        :param target_transformer: таргет трансформер.
        :param x: Данные для получения предикта.
        :return:
        """
        eval_batch_size = 8096
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = scaler.transform(x.astype(np.float32))
        x = torch.as_tensor(x).to(device)
        y_pred: np.ndarray = (
            torch.cat([model(x[idx]).squeeze(-1).float()
                       for idx in torch.arange(x.shape[0], device=device).split(eval_batch_size)])
        ).detach().cpu().numpy()
        assert target_transformer is not None
        y_pred = y_pred * target_transformer.std + target_transformer.mean
        y_pred = y_pred.mean(1)
        return y_pred.flatten()

    def get_models_preds(self,
                         base_models: list,
                         scalers: list,
                         target_transformers: list,
                         x: np.ndarray) -> np.ndarray:
        """
        Делает предикт нескольких базовых моделей.

        :param models: Список c базовыми моделями
        :param scalers: Список со скейлерами для базовых моделей.
        :param target_transformers: Список с таргет трансформерами для базовых моделей
        :param x: Данные для получения предиктов
        :return: Предикты базовых моделей
        """
        preds = None
        for i, model in enumerate(base_models):
            try:
                model_preds = model.predict(scalers[i].transform(x))
            except:
                model_preds = self.get_predict_tabm(model, scalers[i], target_transformers[i], x)
            preds = model_preds if preds is None else np.vstack((preds, model_preds))
        return preds.T

    def mean_models_preds(self, models: list, scalers: list, target_transformers: list, x: np.ndarray):
        """
        Усредняет предикты базовых моделей

        :param models: Список с базовыми моделями обученных на разных фолдах.
        :param scalers: Список со скейлерами для базовых моделей для разных фолдов.
        :param target_transformers: Список с таргет трансформерами для базовых моделей для разных фолдов.
        :param x: Данные для получения предсказаний.
        :return: Усредненные предикты по моделям.
        """
        preds = None
        for i in range(len(models)):
            tmp_pred = self.get_models_preds(models[i], scalers[i], target_transformers[i], x)
            preds = tmp_pred if preds is None else preds + tmp_pred
        return preds / len(models)

    def train_final_ensemble(self, learn_linear: bool = LEARN_LINEAR) -> tuple:
        """
        Тренировка итогового ансамбля. Обучаем на 5 фолдах базовые модели.
        Затем поверх них обучаем линейную регрессию или просто возвращает ансамбль алгоритмов

        :param learn_linear: используется ли блендинг.
        :return: Кортеж из списка моделей, скейлеров, таргет трансформеров и блендинг модели.
        """
        print('Старт обучения ансамбля')

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        models = list()
        scalers = list()
        target_transformers = list()

        for i, (train_index, val_index) in enumerate(kf.split(self.train_dataset.features_train)):
            print(f'Обучение на фолде {i}')
            fold_models = list()
            fold_scalers = list()
            fold_target_transformers = list()
            # обучаем ансамбль моделей
            x_train_fold = self.train_dataset.features_train[train_index]
            y_train_fold = self.train_dataset.targets_train[train_index]
            mid_prices_train_fold = self.train_dataset.mid_prices_train[train_index]

            x_val_fold = self.train_dataset.features_train[val_index]
            y_val_fold = self.train_dataset.targets_train[val_index]
            mid_prices_val_fold = self.train_dataset.mid_prices_train[val_index]

            # Обучаем xgboost
            if 'xgb' in ENSEMBLE_MODELS:
                print('Старт обучения xgb')
                model_xgb, scaler_xgb, target_transformer = self.baseline_train(x_train_fold,
                                                                                y_train_fold,
                                                                                x_val_fold,
                                                                                y_val_fold,
                                                                                mid_prices_train_fold,
                                                                                mid_prices_val_fold,
                                                                                algo='xgb')
                fold_models.append(model_xgb)
                fold_scalers.append(scaler_xgb)
                fold_target_transformers.append(None)

            # Обучаем lgbm
            if 'lgbm' in ENSEMBLE_MODELS:
                print('Старт обучения lgbm')
                model_lgmb, scaler_lgbm, target_transformer = self.baseline_train(x_train_fold,
                                                                                  y_train_fold,
                                                                                  x_val_fold,
                                                                                  y_val_fold,
                                                                                  mid_prices_train_fold,
                                                                                  mid_prices_val_fold,
                                                                                  algo='lgbm')
                fold_models.append(model_lgmb)
                fold_scalers.append(scaler_lgbm)
                fold_target_transformers.append(None)

            # Обучаем Tabm
            if 'tabm' in ENSEMBLE_MODELS:
                print('Старт обучения tabm')
                scaler_tabm = StandardScaler()
                x_train_fold = scaler_tabm.fit_transform(x_train_fold)
                x_val_fold = scaler_tabm.transform(x_val_fold)
                model_tabm, target_transformer = \
                    train_tabm(x_train_fold, y_train_fold, x_val_fold, y_val_fold, i)

                fold_models.append(model_tabm)
                fold_scalers.append(scaler_tabm)
                fold_target_transformers.append(target_transformer)

            models.append(fold_models)
            scalers.append(fold_scalers)
            target_transformers.append(fold_target_transformers)

        if learn_linear:
            print('Выбран блендинг с линейной регрессией')
            blending_model = Ridge()
            blending_x_train = None
            blending_y_train = None
            for i, (train_index, val_index) in enumerate(kf.split(self.train_dataset.features_train)):
                x_train_fold = self.train_dataset.features_train[val_index]
                if blending_x_train is None:
                    blending_x_train = self.get_models_preds(models[i], scalers[i], target_transformers[i],
                                                             x_train_fold)
                    blending_y_train = self.train_dataset.targets_train[val_index]
                else:
                    blending_x_train = np.vstack((blending_x_train, self.get_models_preds(models[i],
                                                                                          scalers[i],
                                                                                          target_transformers[i],
                                                                                          x_train_fold)))
                    blending_y_train = np.vstack((blending_y_train, self.train_dataset.targets_train[val_index]))
            blending_model.fit(blending_x_train, blending_y_train.reshape(-1))
            print(f'Веса базовых алгоритмов в блендинге: {blending_model.coef_}')
            blending_preds = blending_model.predict(self.mean_models_preds(models, scalers, target_transformers,
                                                                           self.train_dataset.features_test))
            get_stats(blending_preds, self.train_dataset.targets_test, self.train_dataset.mid_prices_test, 'Test')
        else:
            print('Выбрано обычное усреднение моделей')
            blending_model = None
            ensemble_preds = self.predict_ensemble(models, scalers, target_transformers, blending_model,
                                                   self.train_dataset.features_test)
            ensemble_accuracy = calc_accuracy_score(self.train_dataset.targets_test, ensemble_preds,
                                                    self.train_dataset.mid_prices_test)
            print(f'Качество ансамбля на отложенной выборке: {ensemble_accuracy}')
        return models, scalers, target_transformers, blending_model

    def predict_ensemble(self,
                         models: list,
                         scalers: list,
                         target_transformers: list,
                         blending_model: list,
                         features: np.ndarray) -> np.array:
        """
        Получение предиктов ансамбля.

        :param models: Список базовых моделей по фолдам.
        :param scalers: Список скейлеров для базовых моделей.
        :param target_transformers: Список таргет трансформеров для базовых моделей.
        :param blending_model: Блендинг модель
        :param features: Данные для формирования предикта.
        :return: Предикты.
        """
        if blending_model:
            return None
        else:
            ensemble_preds = None
            for model_number in range(len(models[0])):
                print(f'model_number : {model_number}')
                model_preds = None
                for fold in range(N_FOLDS):
                    print(f'fold : {fold}')
                    model = models[fold][model_number]
                    scaler = scalers[fold][model_number]
                    target_transformer = target_transformers[fold][model_number]
                    try:
                        preds = model.predict(scaler.transform(features))
                    except:
                        preds = self.get_predict_tabm(model, scaler, target_transformer,
                                                      features)
                    model_preds = preds if model_preds is None else np.vstack((model_preds, preds))
                model_preds = np.mean(model_preds, axis=0)
                print(f'Оценка модели {model_number} на отложенной выборке')
                ensemble_preds = model_preds if ensemble_preds is None else np.vstack((ensemble_preds, model_preds))
            if len(models[0]) > 1:
                ensemble_preds = np.mean(ensemble_preds, axis=0)
        return ensemble_preds

    def make_final_submission_model(self, model: Any, scaler: Any):
        """
        Формирование финального сабмита моделью.

        :param model: модель.
        :param scaler: скейлер для предобработки данных.
        :return:
        """
        self.test_dataset.build_final_test()
        preds = model.predict(scaler.transform(self.test_dataset.features))
        self.test_dataset.final_test_preds = preds
        self.test_dataset.save_final_result()
        return

    def make_final_submission_ensemble(self,
                                       models: list,
                                       scalers: list,
                                       target_transformers: list,
                                       blending_model: Any):
        """
        Формирование финального сабмита ансамблем.

        :param models: Список с базовыми моделями по фолдам.
        :param scalers: Скейлеры.
        :param target_transformers: Таргет трансформеры.
        :param blending_model: Блендинг модель
        :return:
        """
        self.test_dataset.build_final_test()
        preds = self.predict_ensemble(models, scalers, target_transformers, blending_model, self.test_dataset.features)
        self.test_dataset.final_test_preds = preds
        self.test_dataset.save_final_result()
        return
