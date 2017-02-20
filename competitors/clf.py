import math
import datetime
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


def get_bin_id(days_since_last_bought):
    bin_id = int(days_since_last_bought) / 5

    return min(bin_id, 10)


def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))


class CLF:

    def __init__(self, min_item_occurrences=1):

        self.__state = 'initialized'
        self.min_item_occurrences = min_item_occurrences

        self.item_clf = None
        self.item_last_basket_features = None
        self.last_basket = None

    def get_state(self):
        return self.__state

    def build_model(self, baskets):
        self.__state = 'built'

        sorted_basket_ids = sorted(baskets['data'])

        items_shop_features = dict()
        items_date_last_purchase = dict()
        items_count_purchases = dict()
        items_freq_histogram = dict()

        # calculates temporal features for each item
        for basket_id in sorted_basket_ids:
            date_object = datetime.datetime.strptime(basket_id[0:10], '%Y_%m_%d')

            basket = baskets['data'][basket_id]['basket'].keys()
            for item in items_shop_features:
                days_since_last_bought = 1.0 * (date_object - items_date_last_purchase[item]).days
                dow = date_object.weekday()
                hour = date_object.hour / 6
                month = date_object.month
                quarter = (month / 4) + 1

                item_purchased = 0
                if item in basket:
                    item_purchased = 1
                    items_count_purchases[item] += 1
                    items_date_last_purchase[item] = date_object

                    bin_id = get_bin_id(days_since_last_bought)
                    items_freq_histogram[bin_id] += 1.0

                bin_id = get_bin_id(days_since_last_bought)
                frequency_of_interval = items_freq_histogram[bin_id] / items_count_purchases[item]

                min1_trans = [0, 0, 0, 0, 0, 0, 0]
                min2_trans = [0, 0, 0, 0, 0, 0, 0]
                min3_trans = [0, 0, 0, 0, 0, 0, 0]
                min4_trans = [0, 0, 0, 0, 0, 0, 0]

                if len(items_shop_features[item]) > 0:
                    min1_trans = items_shop_features[item][-1][:7]
                if len(items_shop_features[item]) > 1:
                    min2_trans = items_shop_features[item][-2][:7]
                if len(items_shop_features[item]) > 2:
                    min3_trans = items_shop_features[item][-3][:7]
                if len(items_shop_features[item]) > 3:
                    min4_trans = items_shop_features[item][-4][:7]

                item_basket_features = [days_since_last_bought,
                                        frequency_of_interval,
                                        bin_id,
                                        dow,
                                        hour,
                                        month,
                                        quarter]

                item_basket_features.extend(min1_trans)
                item_basket_features.extend(min2_trans)
                item_basket_features.extend(min3_trans)
                item_basket_features.extend(min4_trans)

                # da mettere se si usa un linear classifier tipo perceptron
                # non_linear_combinations = list()
                # for i in xrange(7):
                #     values = list()
                #     for j in xrange(5):
                #         values.append(item_basket_features[i + j * 7])
                #     if i < 2:
                #         non_linear_combinations.append(sigmoid(np.sum(values)))
                #     else:
                #         val = 1.0
                #         for j in xrange(4):
                #             if values[j] != values[j + 1]:
                #                 val = 0.0
                #                 break
                #         non_linear_combinations.append(val)
                #
                # item_basket_features.extend(non_linear_combinations)

                item_basket_features.extend([item_purchased])

                items_shop_features[item].append(item_basket_features)

            for item in basket:
                if item not in items_shop_features:
                    items_shop_features[item] = list()
                    items_date_last_purchase[item] = date_object
                    items_count_purchases[item] = 1
                    items_freq_histogram = {
                        0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
                        6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0
                    }

        self.last_basket = baskets['data'][sorted_basket_ids[-1]]['basket'].keys()
        self.item_clf = dict()
        self.item_last_basket_features = dict()

        # train classifier for each item
        for item in items_shop_features:
            X_train = list()
            Y_train = list()

            for features in items_shop_features[item]:
                # X_train.append(features[:-1]) caso perceptron
                X_train.append(features[:-7])
                Y_train.append(features[-1])

            if np.sum(Y_train) <= self.min_item_occurrences - 1:
                continue

            clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                         min_samples_split=20, min_samples_leaf=10,
                                         min_weight_fraction_leaf=0.0, max_features=None,
                                         random_state=None, max_leaf_nodes=None,
                                         min_impurity_split=1e-07, class_weight=None, presort=False)

            clf.fit(X_train, Y_train)
            self.item_clf[item] = clf
            self.item_last_basket_features[item] = np.reshape(X_train[-1], (1, -1))

        return self

    def update(self, new_baskets):
        if self.__state != 'built':
            raise Exception('Model not built, cannot be updated')
        return self.build_model(new_baskets)

    def predict(self, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        item_score = dict()
        for item in self.item_clf:
            clf = self.item_clf[item]
            prediction = clf.predict_proba(self.item_last_basket_features[item])
            if len(prediction[0]) == 1:
                will_be_purchased = clf.predict(self.item_last_basket_features[item])[0]
                if will_be_purchased == 1:
                    item_score[item] = 1.0
                else:
                    item_score[item] = 0.0
            else:
                item_score[item] = prediction[0][1]

        if len(item_score) > 0:
            max_nbr_item = min(pred_length, len(item_score))
            pred_basket = sorted(item_score, key=item_score.get, reverse=True)[:max_nbr_item]
        else:
            pred_basket = self.last_basket

        return pred_basket


def main():
    print 'CLF Competitor Test'

    pred_length = 5
    dataset = 'coop'

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
        item2category = None
    else:
        print 'Unnown dataset'
        return

    print datetime.datetime.now(), 'Customers', len(customers_data)

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
    customers_clf_recsys = dict()
    start_time = datetime.datetime.now()
    for i, customer_id in enumerate(customers_train_set):
        # print datetime.datetime.now(), 1.0*i/len(customers_train_set)
        customer_train_set = customers_train_set[customer_id]

        clf = CLF(min_item_occurrences=2)
        clf.build_model(customer_train_set)

        customers_clf_recsys[customer_id] = clf
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        # if customer_id not in customers_clf_recsys:
        #     continue

        clf = customers_clf_recsys[customer_id]
        pred_basket = clf.predict(pred_length)
        pred_basket = set(pred_basket)

        customer_test_set = customers_test_set[customer_id]['data']

        for next_basket_id in customer_test_set:
            next_basket = customer_test_set[next_basket_id]['basket'].keys()
            next_basket = set(next_basket)
            # print pred_basket, next_basket
            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Prediction performed in', end_time - start_time

    f1_values = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print datetime.datetime.now(), 'CLF', 'avg', stats['avg']
    print datetime.datetime.now(), 'CLF', 'iqm', stats['iqm']

    # for k, v in stats.iteritems():
    #     print k, v


if __name__ == "__main__":
    main()
