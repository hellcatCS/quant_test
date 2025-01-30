"""Парсинг данных в датасет"""

from config import LEN_ORDERBOOK
from utils.dataset import Dataset
from utils.models import Order, OrderBook


def create_dataset(data_path: str, mode: str = 'Train') -> Dataset:
    """
    Парсит данные из txt файла в датасет

    :param data_path: Путь к данным.
    :param mode: Трейн или тест датасет.
    """
    dataset = Dataset(mode=mode)
    orderbook = OrderBook()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line[0] == '=':
                assert len(orderbook.orders) == LEN_ORDERBOOK
                dataset.orderbooks.append(orderbook)
                orderbook = OrderBook()
            else:
                params = line.split()
                if line[0].isdigit():
                    side = 1 if params[-1] == 'Buy' else -1
                    order = Order(price=int(params[0]), amount=int(params[1]), side=side)
                    orderbook.orders.append(order)
                else:
                    side = 1 if params[0] == 'Buy' else -1
                    orderbook.predict_side = side
                    if mode == 'Train':
                        orderbook.predict_price = params[2]
                    else:
                        orderbook.predict_price = None
    print(f'Инициализирован {mode} датасет, с количество стаканов: {len(dataset.orderbooks)}')
    return dataset
