from tars import *

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


class TBP:

    def __init__(self):
        self.__state = 'initialized'
        self.tars_tree = None
        self.tars = None
        self.rs_intervals_support = None

    def get_state(self):
        return self.__state

    def build_model(self, baskets):

        self.tars_tree = TARSTree(baskets, root_value=None, root_count=None, root_timeseries=None)
        self.tars = self.tars_tree.mine_patterns(max_rec_dept=0, patterns_subset=None, nbr_patterns=None,
                                                 get_items_in_order_of_occurrences=True)
        self.nbr_patterns = len(self.tars)
        self.rs_intervals_support = calculate_intervals_support(self.tars, self.tars_tree)
        self.__state = 'built'

        return self

    def update_model(self, new_baskets):
        return self.build_model(new_baskets)

    def predict(self, customer_data, day_of_next_purchase, nbr_patterns, pred_length=5, queue=None):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        if nbr_patterns is not None and nbr_patterns > 0:
            rs_purchases, rs_day_of_last_purchase = calcualte_active_rp(customer_data, self.rs_intervals_support,
                                                                        day_of_next_purchase)

            self.tars = self.tars_tree.mine_patterns(max_rec_dept=1, patterns_subset=rs_purchases,
                                                     nbr_patterns=nbr_patterns,
                                                     get_items_in_order_of_occurrences=False)
            self.rs_intervals_support = calculate_intervals_support(self.tars, self.tars_tree)

        rs_purchases, rs_day_of_last_purchase = calcualte_active_rp(customer_data, self.rs_intervals_support,
                                                                        day_of_next_purchase)

        self.nbr_active_patterns = len(rs_purchases)

        item_score = calcualte_item_score(self.tars_tree, rs_purchases, self.rs_intervals_support)

        max_nbr_item = min(pred_length, len(item_score))
        pred_basket = sorted(item_score, key=item_score.get, reverse=True)[:max_nbr_item]

        if queue is not None:
            queue.put(pred_basket)

        return pred_basket
