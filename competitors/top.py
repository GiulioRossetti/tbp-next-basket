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

        # if pred_length > self.n_items:
        #     raise Exception('pred_length higher than n_items, (%s > %s)' % (pred_length, self.n_items))
        #
        # pred_basket = list()
        # max_nbr_item = min(pred_length, len(self.top_items))
        # for i in range(0, max_nbr_item):
        #     pred_basket.append(self.top_items[i])

        max_nbr_item = min(pred_length, len(self.item_count))
        pred_basket = sorted(self.item_count, key=self.item_count.get, reverse=True)[:max_nbr_item]

        return pred_basket


def main():
    print 'Top Competitor Test'

    pred_length = 5
    dataset = 'tmall'

    print datetime.datetime.now(), 'Read dataset', dataset

    if dataset == 'tafeng':
        path = '/Users/riccardo/Documents/PhD/NextBasket/Competitors/Dataset/TaFeng/D11-02/'
        customers_data = read_data(path + 'tafeng.json')
        item2category = None
    elif dataset == 'tmall':
        path = '/Users/riccardo/Documents/PhD/NextBasket/Competitors/Dataset/Tmall/'
        customers_data = read_data(path + 'tmall.json')
        item2category = None
    elif dataset == 'coop':
        path = '/Users/riccardo/Documents/PhD/NextBasket/Dataset/'
        # customers_data = read_data(path + 'dataset_livorno_prov_filtered_iqr_head2500_ok.json')
        customers_data = read_data(path + 'dataset100.json')
        item2category = get_item2category(path + 'coop_categories_map.csv', category_index['categoria'])
        # item2category = None
    else:
        print 'Unnown dataset'
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

    print datetime.datetime.now(), 'Customers for test', len(customers_train_set), \
        '%.2f%%' % (100.0*len(customers_train_set)/len(customers_data))

    print datetime.datetime.now(), 'Create and build models'
    customers_top_recsys = dict()
    baskets_all = list()
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set.keys()[:100]:
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)

        top = Top()
        top.build_model(baskets)

        customers_top_recsys[customer_id] = top

        baskets_all.extend(baskets)
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    # top_all = Top()
    # top_all.build_model(baskets_all)
    # pred_all = top_all.predict(pred_length)
    # pred_all = set(pred_all)
    # end_time = datetime.datetime.now()
    # print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    performances_all = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        if customer_id not in customers_top_recsys:
            continue
        top = customers_top_recsys[customer_id]
        pred_basket = top.predict(pred_length)
        pred_basket = set(pred_basket)

        customer_test_set = customers_test_set[customer_id]
        next_baskets = data2baskets(customer_test_set)

        for next_basket in next_baskets:
            next_basket = set(next_basket)
            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)

            # evaluation_all = evaluate_prediction(next_basket, pred_all)
            # performances_all[customer_id].append(evaluation_all)
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Prediction performed in', end_time - start_time

    f1_values = list()
    f1_values_all = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    for customer_id in performances_all:
        for evaluation_all in performances_all[customer_id]:
            f1_values_all.append(evaluation_all['f1_score'])

    stats = calculate_aggregate(f1_values)
    print datetime.datetime.now(), 'P-TOP', stats['avg']
    print datetime.datetime.now(), 'P-TOP', stats['iqm']

    # stats_all = calculate_aggregate(f1_values_all)
    # print datetime.datetime.now(), 'NP-TOP', stats_all['avg']
    # print datetime.datetime.now(), 'NP-TOP', stats_all['iqm']

    # for k, v in stats.iteritems():
    #     print k, v


if __name__ == "__main__":
    main()


