"""Рассчет основного пайплайна"""

import random
import time

import numpy as np
from sklearn.model_selection import train_test_split

from config import PATH_TO_TRAIN, PATH_TO_TEST, SEED, VAL_SIZE, MODE
from utils.parsing import create_dataset
from utils.solver import Solver

random.seed(SEED)
np.random.seed(SEED)

start_time = time.perf_counter()

train_dataset = create_dataset(PATH_TO_TRAIN)
test_dataset = create_dataset(PATH_TO_TEST, mode='Test')

solver = Solver(train_dataset, test_dataset)

if MODE == 'model':
    print('Выбран режим одной модели')
    x_train, x_val, y_train, y_val = train_test_split(solver.train_dataset.features_train,
                                                      solver.train_dataset.targets_train,
                                                      test_size=VAL_SIZE,
                                                      random_state=SEED)
    mid_train, mid_val = train_test_split(solver.train_dataset.mid_prices_train,
                                          test_size=VAL_SIZE,
                                          random_state=SEED)
    model, scaler, transformer = solver.baseline_train(x_train, y_train, x_val, y_val, mid_train, mid_val)
    solver.predict_on_hold_out(model, scaler)
    solver.make_final_submission_model(model, scaler)
elif MODE == 'ensemble':
    print('Выбран режим ансамбля')
    models, scalers, target_transformers, blending_model = solver.train_final_ensemble()
    solver.make_final_submission_ensemble(models, scalers, target_transformers, blending_model)

end_time = time.perf_counter()

print(f'Время выполнения: {end_time - start_time}')
