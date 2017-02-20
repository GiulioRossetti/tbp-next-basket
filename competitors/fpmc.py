import math
import random
import numpy as np

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


def calculate_baskets_for_drawing(baskets):
    basket_for_drawing = list()
    for u, user_baskets in enumerate(baskets):
        for t, basket in enumerate(user_baskets):
            for i in basket:
                basket_for_drawing.append((u, t, i))
    return basket_for_drawing


def calculate_probabilities_for_drawing(baskets):
    user_items_count = defaultdict(float)
    user_basket_count = defaultdict(lambda: defaultdict(float))

    tot_purchases = 0.0
    for u, user_baskets in enumerate(baskets):
        for t, basket in enumerate(user_baskets):
            for i in basket:
                user_items_count[u] += 1.0
                user_basket_count[u][t] += 1.0
                tot_purchases += 1.0

    user_probability = dict()
    user_basket_probability = defaultdict(dict)
    for u in user_items_count:
        user_probability[u] = user_items_count[u] / tot_purchases
        for t in user_basket_count[u]:
            user_basket_probability[u][t] = user_basket_count[u][t] / np.sum(user_basket_count[u].values())

    return user_probability, user_basket_probability


# def draw_uti(baskets, user_probability, user_basket_probability):
#     u = np.random.choice(user_probability, 1, p=[user_probability[k] for k in sorted(user_probability)])[0]
#     t = np.random.choice(range(0, len(baskets[u])), 1,
#                          p=[user_basket_probability[u][k] for k in sorted(user_basket_probability[u])])[0]
#     i = random.choice(baskets[u][t])
#
#     return u, t, i


def draw_uti(basket_for_drawing):
    return random.choice(basket_for_drawing)


def compute_x(u, i, basket, VUI, VIU, VIL, VLI):
    x_ui = np.dot(VUI[u], VIU[i])
    x_il = 0.0
    for l in basket:
        x_il += np.dot(VIL[i], VLI[l])
    x_il /= len(basket)
    x = x_ui + x_il
    return x


def sigmoid(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))


class FPMC:

    def __init__(self, n_user, n_item, n_factor, alpha=0.01, lambdas=(0.001, 0.001, 0.001, 0.001), std=0.01,
                 verbose=False):
        self.__state = 'initialized'

        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor

        self.alpha = alpha
        self.lambdaUI = lambdas[0]
        self.lambdaIU = lambdas[1]
        self.lambdaIL = lambdas[2]
        self.lambdaLI = lambdas[3]
        self.std = std
        self.verbose = verbose

        self.n_epoch = None
        self.tolerance = None
        self.convergence = False
        self.VUI = None
        self.VIU = None
        self.VIL = None
        self.VLI = None
        self.I = None

        self.basket_for_drawing = None

    def get_state(self):
        return self.__state

    def __learn_epoch(self, baskets):

        for iter_idx in xrange(len(baskets)):
            u, t, i = draw_uti(self.basket_for_drawing)

            basket_tm1 = baskets[u][t]
            j = random.choice(list(self.I - set(basket_tm1)))
            x_uti = compute_x(u, i, basket_tm1, self.VUI, self.VIU, self.VIL, self.VLI)
            x_utj = compute_x(u, j, basket_tm1, self.VUI, self.VIU, self.VIL, self.VLI)

            delta = (1 - sigmoid(x_uti - x_utj))

            VUI_update = self.alpha * (delta * (self.VIU[i] - self.VIU[j]) - self.lambdaUI * self.VUI[u])
            VIUi_update = self.alpha * (delta * self.VUI[u] - self.lambdaIU * self.VIU[i])
            VIUj_update = self.alpha * (-delta * self.VUI[u] - self.lambdaIU * self.VIU[j])

            self.VUI[u] += VUI_update
            self.VIU[i] += VIUi_update
            self.VIU[j] += VIUj_update

            eta = np.mean(self.VLI[basket_tm1], axis=0)

            VILi_update = self.alpha * (delta * eta - self.lambdaIL * self.VIL[i])
            VILj_update = self.alpha * (-delta * eta - self.lambdaIL * self.VIL[j])
            VLI_update = self.alpha * ((delta * (self.VIL[i] - self.VIL[j]) / len(basket_tm1)) -
                                       self.lambdaLI * self.VLI[basket_tm1])

            self.VIL[i] += VILi_update
            self.VIL[j] += VILj_update
            self.VLI[basket_tm1] += VLI_update

            if np.all(np.abs(VUI_update) < self.tolerance) and \
                np.all(np.abs(VIUi_update) < self.tolerance) and \
                np.all(np.abs(VIUj_update) < self.tolerance) and \
                np.all(np.abs(VILi_update) < self.tolerance) and \
                np.all(np.abs(VILj_update) < self.tolerance) and \
                np.all(np.abs(VLI_update) < self.tolerance):
                self.convergence = True
                break

    def __learn_sbpr_fpmc(self, baskets):
        self.VUI = np.random.normal(0.0, self.std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0.0, self.std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0.0, self.std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0.0, self.std, size=(self.n_item, self.n_factor))

        self.basket_for_drawing = calculate_baskets_for_drawing(baskets)

        for epoch in xrange(self.n_epoch):
            if self.verbose:
                print datetime.datetime.now(), 'Epoch %s' % epoch
            self.__learn_epoch(baskets)
            if self.convergence:
                break

        self.VUI_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_VLI = np.dot(self.VIL, self.VLI.T)

    def build_model(self, baskets, items, n_epoch=10, tolerance=1.0e-8):
        self.I = items
        self.n_epoch = n_epoch
        self.tolerance = tolerance
        self.__learn_sbpr_fpmc(baskets)
        self.__state = 'built'

        return self

    def update_model(self, new_baskets):
        self.build_model(new_baskets, self.I)
        return self

    def predict(self, user_id, last_basket, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        scores = self.VUI_VIU[user_id] + np.mean(self.VIL_VLI[:, last_basket], axis=1)
        item_rank = {k: v for k, v in enumerate(scores)}

        max_nbr_item = min(pred_length, len(item_rank))
        pred_basket = sorted(item_rank, key=item_rank.get, reverse=True)[:max_nbr_item]

        return pred_basket

# baskets = [
#     [['a', 'b', 'c'], ['b', 'c'], ['a', 'b']],
#     [['a'], ['a', 'c']],
#     [['d'], ['c', 'e'], ['e']],
#     [['c', 'e']]
# ]
#
# new_baskets, new2old, old2new = remap_items(baskets)
# baskets = new_baskets
# items = get_items(baskets)
# items = set(items.keys())
#
# fpmc = FPMC(len(baskets), len(items), 4)
# fpmc.build_model(baskets, items, 1)
#
# pred_basket = fpmc.predict(0, baskets[0][len(baskets[0]) - 1])
#
# print pred_basket
# print fpmc.convergence


def main():
    print 'FPMC Competitor Test'

    pred_length = 5
    n_factor = 200

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
        #item2category = None
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

    customers_baskets = list()
    userid_customerid = dict()
    customerid_userid = dict()
    print datetime.datetime.now(), 'Create and build models'
    start_time = datetime.datetime.now()
    for user_id, customer_id in enumerate(customers_train_set):
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)
        customers_baskets.append(baskets)
        userid_customerid[user_id] = customer_id
        customerid_userid[customer_id] = user_id

    customers_baskets, new2old, old2new = remap_items(customers_baskets)
    items = new2old.keys()
    items = set(items)

    fpmc = FPMC(len(customers_baskets), len(items), n_factor,
                alpha=0.01, lambdas=(0.001, 0.001, 0.001, 0.001), std=0.01, verbose=False)
    fpmc.build_model(customers_baskets, items, n_epoch=1000, tolerance=1.0e-8)
    print 'convergence', fpmc.convergence
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)

    # print 'n items', len(items)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        user_id = customerid_userid[customer_id]
        last_basket = customers_baskets[user_id][len(customers_baskets[user_id]) - 1]
        pred_basket = fpmc.predict(user_id, last_basket, pred_length)
        pred_basket = set([new2old[item] for item in pred_basket])

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
    print datetime.datetime.now(), 'FPMC', 'avg', stats['avg']
    print datetime.datetime.now(), 'FPMC', 'iqm', stats['iqm']

    # print stats


if __name__ == "__main__":
    main()
