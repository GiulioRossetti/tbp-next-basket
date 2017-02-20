from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


class Top:

    def __init__(self, n_items=10):
        self.n_items = n_items
        self.__state = 'initialized'
        self.item_count = None
        self.top_items = None

    def get_state(self):
        return self.__state

    def __count_items(self, baskets):

        for basket in baskets:
            for item_id in basket:
                self.item_count[item_id] += 1

        # top_item_idx = min(self.n_items, len(self.item_count))
        # self.top_items = sorted(self.item_count, key=self.item_count.get, reverse=True)[:top_item_idx]

    def build_model(self, baskets):
        if self.__state != 'initialized':
            raise Exception('Model already modified, cannot be re-built')

        self.__state = 'built'
        self.item_count = defaultdict(int)
        self.__count_items(baskets)
        return self

    def update_model(self, new_baskets):
        if self.__state != 'built':
            raise Exception('Model not built, cannot be updated')

        self.__count_items(new_baskets)
        return self

    def predict(self, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        max_nbr_item = min(pred_length, len(self.item_count))
        pred_basket = sorted(self.item_count, key=self.item_count.get, reverse=True)[:max_nbr_item]

        return pred_basket
