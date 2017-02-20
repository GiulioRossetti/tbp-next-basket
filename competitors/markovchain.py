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

        # for item_l in self.probability_matrix:
        #     for item_i in self.probability_matrix[item_l]:
        #         print item_l, item_i, self.probability_matrix[item_l][item_i]
        return self

    def update_model(self, new_baskets):
        if self.__state != 'built':
            raise Exception('Model not built, cannot be updated')
        self.__calculate_proabilities(new_baskets)
        self.last_basket = new_baskets[len(new_baskets) - 1]

        # print 'lb', self.last_basket
        # for item_l in self.probability_matrix:
        #     for item_i in self.probability_matrix[item_l]:
        #         print item_l, item_i, self.probability_matrix[item_l][item_i]
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

# baskets = [
#     ['a', 'b', 'c'],
#     ['b', 'c'],
#     ['a', 'b'],
# ]
#
# # baskets = [
# #     ['d'],
# #     ['c', 'e'],
# #     ['e'],
# # ]
#
# mc = MarkovChain()
# mc.build_model(baskets)
# # print ''
# # mc.update_model([['a'], ['a', 'c']])
# # mc.update_model([['d'], ['c', 'e'], ['e']])
# # print ''
# # mc.update_model([['c', 'e']])
#
#
# pred_basket = mc.predict()
#
# print pred_basket


def main():
    print 'Markov Chain Competitor Test'

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
    for customer_id in customers_train_set.keys()[:100]:
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)
        mc = MarkovChain()
        mc.build_model(baskets)

        customers_recsys[customer_id] = mc
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        if customer_id not in customers_recsys:
            continue
        mc = customers_recsys[customer_id]
        pred_basket = mc.predict(pred_length)
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
    print datetime.datetime.now(), 'MC', 'avg', stats['avg']
    print datetime.datetime.now(), 'MC', 'iqm', stats['iqm']


if __name__ == "__main__":
    main()





