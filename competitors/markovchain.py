from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


class MarkovChain:

    def __init__(self):
        self.__state = 'initialized'
        self.probability_matrix = None
        self.couples_count = None
        self.item_count = None
        self.last_basket = None

    def get_state(self):
        return self.__state

    def __calculate_proabilities(self, baskets):

        for t0 in range(0, len(baskets)-1):
            t1 = t0 + 1
            for item_l in baskets[t0]:
                for item_i in baskets[t1]:
                    self.couples_count[item_l][item_i] += 1.0
                self.item_count[item_l] += 1.0

        self.probability_matrix = defaultdict(lambda: defaultdict(float))
        for item_l in self.couples_count:
            for item_i in self.couples_count[item_l]:
                self.probability_matrix[item_l][item_i] = self.couples_count[item_l][item_i] / self.item_count[item_l]

    def build_model(self, baskets):
        if self.__state != 'initialized':
            raise Exception('Model already modified, cannot be re-built')

        self.couples_count = defaultdict(lambda: defaultdict(float))
        self.item_count = defaultdict(int)
        self.__calculate_proabilities(baskets)
        self.last_basket = baskets[len(baskets) - 1]
        self.__state = 'built'

        return self

    def update_model(self, new_baskets):
        if self.__state != 'built':
            raise Exception('Model not built, cannot be updated')
        self.__calculate_proabilities(new_baskets)
        self.last_basket = new_baskets[len(new_baskets) - 1]

        return self

    def predict(self, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        item_rank = defaultdict(float)
        den = 1.0 * len(self.last_basket)
        for item_i in self.item_count:
            num = 0.0
            for item_l in self.last_basket:
                if item_l in self.probability_matrix:
                    num += self.probability_matrix[item_l].get(item_i, 0.0)
            item_rank[item_i] = 0.0 if num == 0.0 else num / den

        max_nbr_item = min(pred_length, len(item_rank))
        pred_basket = sorted(item_rank, key=item_rank.get, reverse=True)[:max_nbr_item]

        return pred_basket
