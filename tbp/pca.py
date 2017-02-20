from tars import *

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


class PersonalCartAssistant:

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


def main():
    print 'Personal Cart Assistant Test'

    pred_length = 5
    dataset = 'coop'

    print datetime.datetime.now(), 'Read dataset', dataset

    if dataset == 'tafeng':
        path = '/media/riccardo/data1/NextBasket/Dataset/TaFeng/D11-02/'
        customers_data = read_data(path + 'tafeng.json')
        item2category = None
    elif dataset == 'tmall':
        path = '/media/riccardo/data1/NextBasket/Dataset//Tmall/'
        customers_data = read_data(path + 'tmall.json')
        item2category = None
    elif dataset == 'coop':
        path = '/media/riccardo/data1/NextBasket/Dataset/'
        # customers_data = read_data(path + 'dataset_livorno_prov_filtered_iqr_head2500_ok.json')
        customers_data = read_data(path + 'coop100.json')
        item2category = get_item2category(path + 'coop_categories_map.csv', category_index['categoria'])
        # item2category = None
    else:
        print datetime.datetime.now(), 'Unkown dataset'
        return

    print datetime.datetime.now(), 'Customers', len(customers_data)

    print datetime.datetime.now(), 'Partition dataset into train / test'
    customers_train_set, customers_test_set = split_train_test(customers_data,
                                                               split_mode='loo',
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=2,
                                                               item2category=item2category)

    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print datetime.datetime.now(), 'Customers for test', len(customers_train_set), \
        '%.2f%%' % (100.0*len(customers_train_set)/len(customers_data))

    print datetime.datetime.now(), 'Create and build models'
    customers_recsys = dict()
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set.keys()[:11]:
        print datetime.datetime.now(), customer_id
        customer_train_set = customers_train_set[customer_id]
        pca = PersonalCartAssistant()
        pca.build_model(customer_train_set)
        customers_recsys[customer_id] = pca

    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        if customer_id not in customers_recsys:
            continue

        pca = customers_recsys[customer_id]
        customer_data = customers_train_set[customer_id]['data']
        next_baskets = customers_test_set[customer_id]['data']

        for next_basket_id in next_baskets:
            day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')
            pred_basket = pca.predict(customer_data, day_of_next_purchase, nbr_patterns=None, pred_length=pred_length)
            pred_basket = set([new2old[item] for item in pred_basket])

            next_basket = next_baskets[next_basket_id]['basket']
            next_basket = set(next_basket.keys())

            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)

    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Prediction performed in', end_time - start_time

    f1_values = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print datetime.datetime.now(), 'PCA', 'avg', stats['avg']
    print datetime.datetime.now(), 'PCA', 'iqm', stats['iqm']


if __name__ == "__main__":
    main()
