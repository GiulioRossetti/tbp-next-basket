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


def main():
    print 'Last Competitor Test'

    dataset = 'coop'

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
        print datetime.datetime.now(), 'Unnown dataset'
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
    customers_recsys = dict()
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)

        last = Last()
        last.build_model(baskets)

        customers_recsys[customer_id] = last
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        top = customers_recsys[customer_id]
        pred_basket = top.predict()
        pred_basket = set(pred_basket)

        customer_test_set = customers_test_set[customer_id]
        next_baskets = data2baskets(customer_test_set)

        for next_basket in next_baskets:
            next_basket = set(next_basket)
            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Prediction performed in', end_time - start_time

    f1_values = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print datetime.datetime.now(), 'P-LAST', 'avg', stats['avg']
    print datetime.datetime.now(), 'P-LAST', 'iqm', stats['iqm']

    # print ''
    # stats = calculate_aggregate(f1_values)
    # for k, v in stats.iteritems():
    #     print k, v


if __name__ == "__main__":
    main()
