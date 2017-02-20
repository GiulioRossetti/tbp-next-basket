from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


class Last:

    def __init__(self):
        self.last_basket = None
        self.__state = 'initialized'

    def get_state(self):
        return self.__state

    def build_model(self, baskets):
        if self.__state != 'initialized':
            raise Exception('Model already modified, cannot be re-built')

        self.__state = 'built'
        self.last_basket = baskets[len(baskets) - 1]
        return self

    def update_model(self, new_baskets):
        if self.__state != 'built':
            raise Exception('Model not built, cannot be updated')

        self.last_basket = new_baskets[len(new_baskets) - 1]
        return self

    def predict(self):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        pred_basket = self.last_basket

        return pred_basket
