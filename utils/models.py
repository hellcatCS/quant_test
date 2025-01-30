"""Модели для хранения данных"""

from typing import List


class Order:

    def __init__(self, price: int = None, amount: int = None, side: int = None):
        self.price = price
        self.amount = amount
        self.side = side


class OrderBook:

    def __init__(self):
        self.orders = list()
        self.predict_side = None
        self.predict_price = None
