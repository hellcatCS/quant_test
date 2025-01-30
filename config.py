"""Конфиг"""

PATH_TO_TRAIN = 'data/train.txt'
PATH_TO_TEST = 'data/test.txt'

LEN_ORDERBOOK = 40

SEED = 19

HOLD_OUT_TEST_SIZE = 0.1
VAL_SIZE = 0.1

ALGO = 'lgbm'

OPTUNA = True
OPTUNA_SCORER = 'accuracy'

MODE = 'Final'  # 'Debug'

TIMEOUT = 3600
N_THREADS = 4
N_FOLDS = 2 if MODE == 'Debug' else 10
N_TRIALS = 1 if MODE == 'Debug' else 300  # 100

N_EPOCHS = 1 if MODE == 'Debug' else 1000000

LEARN_LINEAR = False

MODE = 'ensemble'  # 'model'
ENSEMBLE_MODELS = ['lgbm']
