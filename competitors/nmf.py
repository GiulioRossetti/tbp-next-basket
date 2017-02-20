# import nimfa
import numpy as np

from sklearn.decomposition import NMF as SKNMF

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


# baskets = [
#     [['a', 'b', 'c'], ['b', 'c'], ['a', 'b']],
#     [['a'], ['a', 'c']],
#     [['d'], ['c', 'e'], ['e']],
#     [['c', 'e']]
# ]
#
# new_baskets, new2old, old2new = remap_items(baskets)
# baskets = new_baskets
#
# user_count = list()
# users_item_count = list()
# item_count = defaultdict(int)
# for user_basket in baskets:
#
#     user_item_count = defaultdict(int)
#     num_purchases = 0
#     for basket in user_basket:
#         for item in basket:
#             num_purchases += 1.0
#             item_count[item] += 1.0
#             user_item_count[item] += 1.0
#
#     user_count.append(num_purchases)
#     users_item_count.append(user_item_count)
#
# n_users = len(baskets)
# n_items = len(item_count)
#
# V = np.zeros((n_users, n_items))
# for u in xrange(n_users):
#     for i in xrange(n_items):
#         r = users_item_count[u].get(i, 0.0)
#         V[u, i] = r
#
# # V = [
# #          [5,3,0,1],
# #          [4,0,0,1],
# #          [1,1,0,5],
# #          [1,0,0,4],
# #          [0,1,5,4],
# #         ]
# #
# # V = np.matrix(V)
# print V
#
#
# print 'nimfa'
# snmf = nimfa.Snmf(V, seed="random_vcol", rank=2, n_run=1, version='r', #eta=1.0,
#                       beta=1e-4, i_conv=10, w_min_change=0, max_iter=30)
# print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
# fit = snmf()
#
# W = fit.basis()
# H = fit.coef()
# # print W
# # print H
# print W.shape
# print H.shape
#
#
# print ''
# R = W * H
#
# np.set_printoptions(suppress=True)
#
# print R
# print ''
#
# for u in xrange(n_users):
#     scores = list(R[u, :].A1)
#     item_rank = {k: v for k, v in enumerate(scores)}
#
#     max_nbr_item = min(5, len(item_rank))
#     pred_basket = sorted(item_rank, key=item_rank.get, reverse=True)[:max_nbr_item]
#
#     print u, pred_basket
#
# # V_actual = np.ma.masked_equal(V, 0)
# # missing_mask = np.ma.getmaskarray(V_actual)
# # R_actual = np.ma.masked_array(R, mask=missing_mask)
# # fit_error = R_actual - V_actual
# # print fit_error
# # fit_error_filled = fit_error.filled(-999)
# # print fit_error_filled
# # actual_ratings = np.where(fit_error_filled > -999)
# # for a in actual_ratings:
# #     print a
# # fit_diffs = np.asarray(fit_error[actual_ratings])
# # print fit_diffs
# # fit_RMSE = np.sqrt(np.sum(fit_diffs ** 2) / fit_diffs.size)
# # print fit_RMSE
#
# # R = np.zeros((n_users, n_items))
# # for u in xrange(n_users):
# #     for i in xrange(n_items):
# #         R[u, i] = (W[u, :] * H[:, i])[0, 0]


class NMF:

    def __init__(self, n_user, n_item, n_factor, alpha=0, l1_ratio=0, beta=1, max_iter=200, tol=1e-4):
        self.__state = 'initialized'

        self.n_users = n_user
        self.n_items = n_item
        self.n_factor = n_factor

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

        self.n_users = None
        self.n_items = None
        self.V = None
        self.W = None
        self.H = None
        self.R = None

    def get_state(self):
        return self.__state

    def __buildV(self, baskets, use_probabilities=False):

        user_count, item_count, users_item_count = count_users_items(baskets)

        self.n_users = len(user_count)
        self.n_items = len(item_count)

        self.V = np.zeros((self.n_users, self.n_items))
        for u in xrange(self.n_users):
            for i in xrange(self.n_items):
                r = users_item_count[u].get(i, 0.0)
                if use_probabilities:
                    r /= user_count[u]
                self.V[u, i] = r

    def build_model(self, baskets, use_probabilities=False):
        # print 'build V'
        self.__buildV(baskets, use_probabilities)
        # print 'density', 1.0 * len(self.V.nonzero()[0]) / (self.V.shape[0] * self.V.shape[1])

        sknmf = SKNMF(n_components=self.n_factor, init='random', solver='cd', tol=self.tol, max_iter=self.max_iter,
                      alpha=self.alpha, l1_ratio=self.l1_ratio, beta=self.beta)

        self.W = sknmf.fit_transform(self.V)
        self.H = sknmf.components_
        self.R = np.dot(self.W, self.H)

        # # print 'nmf'
        # snmfa = nimfa.Snmf(self.V, seed=self.seed, rank=self.rank, n_run=self.n_run, version=self.version,  # eta=1.0,
        #                   beta=self.beta, i_conv=self.i_conv, w_min_change=self.w_min_change, max_iter=self.max_iter)
        # # print 'a'
        # fit = snmfa()
        # # print 'b'
        # self.W = fit.basis()
        # self.H = fit.coef()
        # # print 'mul'
        # self.R = self.W * self.H

        self.__state = 'built'

        return self

    def update_model(self, new_baskets, use_probabilities=False):
        self.build_model(new_baskets, use_probabilities)
        return self

    def predict(self, user_id, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        # scores = list(self.R[user_id, :].A1)
        scores = list(self.R[user_id, :])
        item_rank = {k: v for k, v in enumerate(scores)}

        max_nbr_item = min(pred_length, len(item_rank))
        pred_basket = sorted(item_rank, key=item_rank.get, reverse=True)[:max_nbr_item]

        return pred_basket


def main():
    print 'NMF Competitor Test'

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
        item2category = None
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

    start_time = datetime.datetime.now()
    for user_id, customer_id in enumerate(customers_train_set):
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)
        customers_baskets.append(baskets)
        userid_customerid[user_id] = customer_id
        customerid_userid[customer_id] = user_id

    customers_baskets, new2old, old2new = remap_items(customers_baskets)
    items = new2old.keys()

    print datetime.datetime.now(), 'Create and build models'
    nmf = NMF(len(customers_baskets), len(items), n_factor, alpha=0, l1_ratio=0, beta=1, max_iter=100, tol=1e-4)
    nmf.build_model(customers_baskets, use_probabilities=False)
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        user_id = customerid_userid[customer_id]
        pred_basket = nmf.predict(user_id, pred_length)
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
    print datetime.datetime.now(), 'NMF', 'avg', stats['avg']
    print datetime.datetime.now(), 'NMF', 'iqm', stats['iqm']

    # for k, v in stats.iteritems():
    #     print k, v


if __name__ == "__main__":
    main()



# import nimfa
# import datetime
# import numpy as np
#
# from collections import defaultdict
#
# n_users = 11
# n_items = 1082
#
# start = datetime.datetime.now()
# V = list()
# for u in xrange(n_users):
#     user_vec = [0.0] * n_items
#     for i in xrange(n_items):
#         item_rate = np.random.choice([0,1,2,3,4,5,6,7,8,9], 1, p=[0.3, 0.2, 0.15, 0.15, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
#         user_vec[i] = item_rate[0]
#     V.append(user_vec)
# V = np.matrix(V)
# print 'init', datetime.datetime.now() - start
#
# print V.shape
#
# start = datetime.datetime.now()
# snmf = nimfa.Snmf(V, seed="random_vcol", rank=2, n_run=1, version='r', #eta=1.0,
#                        beta=1e-4, i_conv=10, w_min_change=0, max_iter=30)
#
# fit = snmf()
# print 'fit', datetime.datetime.now() - start
#
#
# start = datetime.datetime.now()
# W = fit.basis()
# H = fit.coef()
# print 'WH', datetime.datetime.now() - start
#
# start = datetime.datetime.now()
# R = W * H
# print 'mul', datetime.datetime.now() - start
