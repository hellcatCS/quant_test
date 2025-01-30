"""Класс датасета"""
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from config import HOLD_OUT_TEST_SIZE, SEED, PATH_TO_TEST
from utils.models import OrderBook


class Dataset:
    """Датасет"""

    def __init__(self, mode: str = 'Train'):
        self.mode = mode
        self.orderbooks = list()
        self.features = None
        self.targets = None
        self.mid_prices = None

        self.features_train = None
        self.targets_train = None
        self.mid_prices_train = None

        self.features_test = None
        self.targets_test = None
        self.mid_prices_test = None

        self.final_test_preds = None

    def build_baseline_feature_vector(self, orderbook: OrderBook) -> Tuple[np.array, float | None, float]:
        """
        Строит вектор признаков из стакана

        :param orderbook: Стакан.
        :return: Возвращает вектор признаков, таргет, а также среднюю цену.
        """
        prices = list()
        amounts = list()
        sides = list()
        for i, order in enumerate(orderbook.orders):
            prices.append(float(order.price))
            amounts.append(float(order.amount))
            sides.append(float(order.side))
        mid_price = (prices[19] + prices[20]) / 2.0
        prices = np.array(prices, dtype='float64')
        prices_sides = prices * np.array(sides)
        normalised_prices = prices - mid_price
        # генерируем основные фичи
        amounts = np.array(amounts)
        feature_vector = normalised_prices * amounts
        side_amounts = amounts * sides
        # Генерируем дополнительные фичи
        sub_features = list()
        # бейзлайн
        sub_features.append(prices[19] - prices[20])  # разница между последним buy и первым sell
        sub_features.append(np.mean(prices))  # среднее в исходных числах
        sub_features.append(orderbook.predict_side)  # buy or sell

        # первая пачка доп фичей
        sub_features.append(prices[0] - prices[39])  # Разница между максимальным и минимальным
        sub_features.append(np.sum(prices[0:20]) - mid_price * 20)
        sub_features.append(mid_price * 20 - np.sum(prices[20:40]))
        sub_features.append(np.sum(amounts))
        sub_features.append(np.sum(amounts[0:20]))
        sub_features.append(np.sum(amounts[20:40]))
        # вторая пачка доп фичей
        sub_features.append(np.sum(feature_vector[0:20]))
        sub_features.append(np.sum(feature_vector[20:40]))
        sub_features.append(amounts[19] - amounts[20])
        # попробуем добавить скользящие средние
        sub_features.append(np.sum(feature_vector[0:5]))
        sub_features.append(np.sum(feature_vector[5:10]))
        sub_features.append(np.sum(feature_vector[10:15]))
        sub_features.append(np.sum(feature_vector[15:20]))
        sub_features.append(np.sum(feature_vector[20:25]))
        sub_features.append(np.sum(feature_vector[25:30]))
        sub_features.append(np.sum(feature_vector[30:35]))
        sub_features.append(np.sum(feature_vector[35:40]))
        # добавим для близких к mid price значениям больше шагов
        sub_features.append(np.sum(feature_vector[21:26]))
        sub_features.append(np.sum(feature_vector[22:27]))
        sub_features.append(np.sum(feature_vector[23:28]))
        sub_features.append(np.sum(feature_vector[24:29]))
        sub_features.append(np.sum(feature_vector[16:21]))
        sub_features.append(np.sum(feature_vector[17:22]))
        sub_features.append(np.sum(feature_vector[18:23]))
        sub_features.append(np.sum(feature_vector[19:24]))
        # попробуем добавить макс и мин статистики
        sub_features.append(np.max(amounts[15:20]))
        sub_features.append(np.max(amounts[20:25]))
        sub_features.append(np.min(amounts[15:20]))
        sub_features.append(np.min(amounts[20:25]))
        # Добавим статистики отношения и разности
        sub_features.append(np.sum(side_amounts[13:20]))
        sub_features.append(np.sum(side_amounts[14:21]))
        sub_features.append(np.sum(side_amounts[15:22]))
        sub_features.append(np.sum(side_amounts[16:23]))
        sub_features.append(np.sum(side_amounts[17:24]))
        sub_features.append(np.sum(side_amounts[18:25]))
        sub_features.append(np.sum(side_amounts[19:26]))
        sub_features.append(np.sum(side_amounts[20:27]))
        # Посчитаем среднюю цену на продажу и покупку
        sub_features.append(np.sum(prices[0:20] * amounts[0:20]) / np.sum(amounts[0:20]))
        sub_features.append(np.sum(prices[20:40] * amounts[20:40]) / np.sum(amounts[20:40]))
        # Считаем стандартные отклонения объемов заказов
        sub_features.append(np.std(amounts[0:40]))
        sub_features.append(np.std(amounts[20:40]))
        sub_features.append(np.std(amounts[0:20]))
        # Отношения количества заказов по окнам
        sub_features.append(np.sum(amounts[19:20]) / np.sum(amounts[20:21]))
        sub_features.append(np.sum(amounts[18:20]) / np.sum(amounts[20:22]))
        sub_features.append(np.sum(amounts[17:20]) / np.sum(amounts[20:23]))
        sub_features.append(np.sum(amounts[16:20]) / np.sum(amounts[20:24]))
        sub_features.append(np.sum(amounts[15:20]) / np.sum(amounts[20:25]))
        sub_features.append(np.sum(amounts[10:20]) / np.sum(amounts[20:30]))
        sub_features.append(np.sum(amounts[0:20]) / np.sum(amounts[20:40]))
        # попробуем исследовать тренд изменения количества ордеров, для начала - дельты
        sub_features.append(amounts[18] - amounts[19])
        sub_features.append(amounts[17] - amounts[18])
        sub_features.append(amounts[16] - amounts[17])
        sub_features.append(amounts[21] - amounts[20])
        sub_features.append(amounts[22] - amounts[21])
        sub_features.append(amounts[23] - amounts[22])
        # попробуем аналогично, только с окнами
        sub_features.append(np.sum(amounts[17:20]) / np.sum(amounts[14:17]))
        sub_features.append(np.sum(amounts[15:20]) / np.sum(amounts[10:15]))
        sub_features.append(np.sum(amounts[10:20]) / np.sum(amounts[0:10]))
        sub_features.append(np.sum(amounts[20:23]) / np.sum(amounts[23:26]))
        sub_features.append(np.sum(amounts[20:25]) / np.sum(amounts[25:30]))
        sub_features.append(np.sum(amounts[20:30]) / np.sum(amounts[30:40]))
        # докинем абсолютные статистики
        sub_features.append(np.max(prices))
        sub_features.append(np.min(prices))
        sub_features.append(np.sum(prices * amounts) / np.sum(amounts))
        # Добавим признаки микротренда
        sub_features.append(int(amounts[19] > amounts[18] > amounts[17]))
        sub_features.append(int(amounts[19] > amounts[18] > amounts[17] > amounts[16]))
        sub_features.append(int(amounts[19] < amounts[18] < amounts[17]))
        sub_features.append(int(amounts[19] < amounts[18] < amounts[17] < amounts[16]))

        sub_features.append(int(amounts[22] > amounts[21] > amounts[20]))
        sub_features.append(int(amounts[23] > amounts[22] > amounts[21] > amounts[20]))
        sub_features.append(int(amounts[22] < amounts[21] < amounts[20]))
        sub_features.append(int(amounts[23] < amounts[22] < amounts[21] < amounts[20]))
        # признаки макротренда
        sub_features.append(int(np.sum(amounts[15:20]) > np.sum(amounts[10:15])))
        sub_features.append(int(np.sum(amounts[15:20]) > np.sum(amounts[10:15]) > np.sum(amounts[5:10])))
        sub_features.append(int(np.sum(amounts[15:20]) < np.sum(amounts[10:15])))
        sub_features.append(int(np.sum(amounts[15:20]) < np.sum(amounts[10:15]) < np.sum(amounts[5:10])))

        sub_features.append(int(np.sum(amounts[20:25]) > np.sum(amounts[25:30])))
        sub_features.append(int(np.sum(amounts[20:25]) > np.sum(amounts[25:30]) > np.sum(amounts[30:35])))
        sub_features.append(int(np.sum(amounts[20:25]) < np.sum(amounts[25:30])))
        sub_features.append(int(np.sum(amounts[20:25]) < np.sum(amounts[25:30]) < np.sum(amounts[30:35])))

        # Добавим отношения скользящих средних взвешенных количеств заказов
        sub_features.append(np.sum(feature_vector[19:20]) / np.sum(feature_vector[20:21]))
        sub_features.append(np.sum(feature_vector[18:20]) / np.sum(feature_vector[20:22]))
        sub_features.append(np.sum(feature_vector[17:20]) / np.sum(feature_vector[20:23]))
        sub_features.append(np.sum(feature_vector[16:20]) / np.sum(feature_vector[20:24]))
        sub_features.append(np.sum(feature_vector[15:20]) / np.sum(feature_vector[20:25]))
        sub_features.append(np.sum(feature_vector[10:20]) / np.sum(feature_vector[20:30]))
        sub_features.append(np.sum(feature_vector[0:20]) / np.sum(feature_vector[20:40]))

        sub_features.append(mid_price)

        feature_vector = np.hstack((feature_vector[10:30], np.array(sub_features)))

        if self.mode == 'Train':
            assert orderbook.predict_price is not None
            return feature_vector, float(orderbook.predict_price), mid_price
        else:
            return feature_vector, None, mid_price

    def build_baseline_features(self):
        """Строит матрицу объекты-признаки для датасета"""
        print('Старт построения фичей для датасета')
        targets = list()
        mid_prices = list()
        for i, orderbook in enumerate(self.orderbooks):
            feature_arr, target, mid_price = self.build_baseline_feature_vector(orderbook)
            if self.features is None:
                self.features = feature_arr
            else:
                self.features = np.vstack((self.features, feature_arr))
            if target is None:
                print(f'Ошибка при построении фичей: на позиции {i} стоит None target')
                raise Exception
            targets.append(target - mid_price)
            mid_prices.append(mid_price)
        self.targets = np.array(targets)
        self.mid_prices = np.array(mid_prices)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.targets,
                                                            test_size=HOLD_OUT_TEST_SIZE, random_state=SEED)
        mid_train, mid_test = train_test_split(self.mid_prices, test_size=HOLD_OUT_TEST_SIZE, random_state=SEED)

        self.features_train = x_train
        self.targets_train = y_train
        self.mid_prices_train = mid_train

        self.features_test = x_test
        self.targets_test = y_test
        self.mid_prices_test = mid_test
        print('Окончание построения фичей для датасета')
        return

    def build_final_test(self):
        """Строит фичи для финального теста"""
        print('Старт построения признаков для итогового теста')
        mid_prices = list()
        for orderbook in self.orderbooks:
            feature_arr, target, mid_price = self.build_baseline_feature_vector(orderbook)
            if self.features is None:
                self.features = feature_arr
            else:
                self.features = np.vstack((self.features, feature_arr))
            mid_prices.append(mid_price)
        self.mid_prices = np.array(mid_prices)
        print('Окончание построение фичей для итогового')

    def save_final_result(self):
        """Сохраняет итоговый ответ"""
        lines = list()
        cnt = 0
        print(self.final_test_preds)
        with open(PATH_TO_TEST, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.split()[0] in ['Sell', 'Buy']:
                    if len(self.final_test_preds) != 25:
                        print(f'Ошибка: количество предиктов на тесте - {len(self.final_test_preds)}')
                    line = line[:-1] + f'{round((self.final_test_preds[cnt] + self.mid_prices[cnt]) / 5) * 5}\n'
                    cnt += 1
                lines.append(line)
        with open("final_preds.txt", "w") as text_file:
            for line in lines:
                text_file.write(line)
