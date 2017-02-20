import math

from dream_model.model.rnnwithbpr_theano_czy import RCnnWithBpr

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


def f1_score5(pre_y, test_y, defult_none):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    for y in pre_y:
        if y != defult_none:
            if y not in set(test_y):
                fp += 1
    for y in test_y:
        if y != -1:
            if y not in set(pre_y):
                fn += 1
            else:
                tp += 1
    return tp, fn, fp


def ndcg5(pre_y, test_y, defult_none):
    length_pre = len(pre_y)
    length_test = 0
    test_y = list(set(test_y))
    for test in test_y:
        if test != -1:
            length_test += 1
    s = 0
    for i in xrange(length_pre):
        if pre_y[i] != defult_none:
            if pre_y[i] in set(test_y):
                if i == 0:
                    s += 1
                else:
                    s += 1.0/math.log(i+1)

    z = 1
    for i in xrange(length_test - 1):
        z += 1.0/math.log(i+2)
    ndcg = s/z
    return ndcg


def prediction_save(folder, name, result):
    for i in xrange(len(result)):
        for j in xrange(len(result[i])):
            result[i][j] = float(result[i][j])
    with open(folder + name + ".json", "w") as f:
        json.dump(result, f)


class DREAM:

    def __init__(self, n_user, n_item, alpha=0.0025, decay=100.0, lambda_r=0.0025, verbose=True,
                 min_nbr_baskets=3, n_hidden=0, embedding_dimension=10, n_epoch=1000, withbpr=True,
                 continue_f=False, isrank=True):

        """
        :param n_user:
        :param n_item:
        :param alpha:
        :param decay:
        :param lambda_r:
        :param verbose:
        :param min_nbr_baskets:
        :param n_hidden: (In the case of bpr embedding is equal to hidden)
        :param embedding_dimension:
        :param n_epoch:
        :param withbpr:
        :param continue_f:
        :param isrank:
        """

        self.n_user = n_user
        self.n_item = n_item

        self.alpha = alpha
        self.decay = decay
        self.lambda_r = lambda_r
        self.verbose = verbose
        self.min_nbr_baskets = min_nbr_baskets
        self.n_hidden = n_hidden
        self.embedding_dimension = embedding_dimension
        self.n_epoch = n_epoch
        self.withbpr = withbpr
        self.continue_f = continue_f
        self.isrank = isrank

        if self.withbpr:
            self.n_hidden = self.embedding_dimension

        self.basket_size = None
        self.recommender = None
        self.__state = 'initialized'

    def __negative(self, users_rank, num_items):
        negtargets = []
        for i in xrange(len(users_rank)):
            negtargets.append([])
            for j in xrange(len(users_rank[i])):
                if users_rank[i][j] == -1:
                    k = -1
                else:
                    k = random.randint(0, num_items - 1)
                    while k in set(users_rank[i]):
                        k = random.randint(0, num_items - 1)
                negtargets[i].append(k)
        return negtargets

    def __get_baskets_info(self, baskets):
        n_basket = 0
        max_basket_len = 0
        tot_nbr_purchases = 0
        baskets_len = list()
        df_items = np.ones(self.n_item)
        pred_num = 1

        for user_baskets in baskets:
            user_baskets_len = list()
            for i, basket in enumerate(user_baskets):
                n_basket += 1
                user_baskets_len.append(len(basket))
                max_basket_len = max(max_basket_len, len(basket))
                tot_nbr_purchases += len(basket)
                for item in basket:
                    df_items[item] += 1.0

                if i == len(user_baskets) - 1:
                    pred_num += len(basket)

            baskets_len.append(user_baskets_len)

        # print 'PRED NUM', pred_num

        self.basket_size = max_basket_len
        self.tot_nbr_purchases = tot_nbr_purchases
        self.baskets_len = baskets_len
        self.df_baskets = np.log(n_basket/df_items)
        self.pred_num = pred_num

    def __tfidf_basket_rank(self, baskets):
        sorted_baskets = list()
        for u, user_baskets in enumerate(baskets):
            sorted_user_baskets = list()
            for b, basket in enumerate(user_baskets):
                tf_baskets = np.zeros(self.n_item)
                items_in_basket = dict()
                for item in basket:
                    tf_baskets[item] += 1.0 / self.tot_nbr_purchases
                    items_in_basket[item] = 0

                tfidf_baskets = self.df_baskets * tf_baskets
                item_rank = {k: v for k, v in enumerate(tfidf_baskets)}
                sorted_basket = sorted(item_rank, key=item_rank.get, reverse=True)
                sorted_basket = [item for item in sorted_basket if item in items_in_basket]
                sorted_user_baskets.append(sorted_basket)
            sorted_baskets.append(sorted_user_baskets)

        return sorted_baskets

    def __pad_data(self, baskets, max_basket_size):

        for u in xrange(len(baskets)):
            for b in xrange(len(baskets[u])):
                while len(baskets[u][b]) < max_basket_size:
                    baskets[u][b].append(-1)
                baskets[u][b] = baskets[u][b][: max_basket_size]

        self.user_basket_dream = baskets

    def build_model(self, baskets, folder):

        self.__state = 'built'

        if self.verbose:
            print("===================RNN Training Model====================\ndata input...")

        self.__get_baskets_info(baskets)
        if self.isrank:
            baskets = self.__tfidf_basket_rank(baskets)
        self.__pad_data(baskets, self.basket_size)  # dizionario di baskets e baskets sizes

        # print 'N ITEMS', self.n_item, self.n_item + 1
        # Create an instance of Rnn
        # Item_size + 1 is to have the last line left to -1, empty items
        self.recommender = RCnnWithBpr(self.basket_size, self.n_hidden, self.n_item + 1, self.embedding_dimension)

        if self.continue_f:
            self.recommender.load_params(folder)

        if self.verbose:
            print("Model was established successfully!\n--------Model training--------\n")

        test_result = []
        better_f1_score = 0
        better_ndcg = 0
        better_epoch_f1 = 0
        better_epoch_ndcg = 0

        for i in xrange(self.n_epoch):

            if self.verbose:
                print datetime.datetime.now(), '\n The %i iteration...' % i

            cost = 0.0

            ndcg = 0.0
            tp = 0.0
            fn = 0.0
            fp = 0.0
            default_num = 0
            result = []

            start_time = datetime.datetime.now()

            if i > 0:
                self.alpha /= math.ceil(i / self.decay)

            # print self.alpha, '<<<<<'

            for user_id in xrange(len(self.user_basket_dream)):
                user_baskets = self.user_basket_dream[user_id]
                user_baskets_sizes = self.baskets_len[user_id]

                # The windowing data itself is handled properly
                # Data collation
                neg_sample = self.__negative(user_baskets, self.n_item)[0:-1]

                train_neg = np.asarray(neg_sample).astype('int32')
                train_x = np.asarray(user_baskets[0:-1]).astype('int32')  # dati
                train_size = np.asarray(user_baskets_sizes[0:-1]).astype('int32')  # dimensione basket

                test_x = np.asarray(user_baskets[0:-1]).astype('int32')
                test_y = np.asarray(user_baskets[-1]).astype('int32')
                test_size = np.asarray(user_baskets_sizes[0:-1]).astype('int32')

                iteration_cost = self.recommender.train(train_x, train_neg, train_size, self.alpha, self.lambda_r)
                cost += iteration_cost

                if i > 0:
                    recall_6_y = self.recommender.evaluation_recall_6(test_x, test_size)
                    if long(self.n_item) in set(recall_6_y):
                        recall_5_y = []
                        default_num += 1
                        for recall in recall_6_y:
                            if long(self.n_item) != recall:
                                recall_5_y.append(recall)
                    else:
                        recall_5_y = recall_6_y[0:-1]
                    result.append(list(recall_5_y))
                    if i < 2:
                        test_result.append(list(test_y))

                    tp_p, fn_p, fp_p = f1_score5(recall_5_y, test_y, self.n_item)
                    tp += tp_p
                    fn += fn_p
                    fp += fp_p
                    ndcg += ndcg5(recall_5_y, test_y, self.n_item)

            # print ''

            # timing display
            if self.verbose:
                print datetime.datetime.now(), 'Epoch %s, loss: %s' % (i, cost), 'Time: ', \
                    datetime.datetime.now()-start_time

            if i > 0:
                f1_score = 2 * tp/(2*tp + fn + fp)
                if i < 2:
                    prediction_save(folder, 'ground_true', test_result)
                if float(f1_score) > better_f1_score:
                        better_f1_score = float(f1_score)
                        better_epoch_f1 = i
                        self.recommender.save_params(folder)
                        better_result = result
                        prediction_save(folder, 'prediction', better_result)
                if float(ndcg) / self.n_user > better_ndcg:
                        better_ndcg = float(ndcg) / self.n_user
                        better_epoch_ndcg = i
                if self.verbose:
                    print '\n cost %.2f\nrecall@5: %f\n F1score@5: %f\n NDCG@: %f' % \
                          (cost, tp / self.pred_num, f1_score, ndcg/self.n_user)
                    print 'The best F1score result appears at the %i iteration: %f' % \
                          (better_epoch_f1, better_f1_score)
                    print 'The best NDCG results appear at the %i iteration and bset_ndcg@5 is:% f' % \
                          (better_epoch_ndcg, better_ndcg)
                    # seeresult()
                    print 'embedding_now max is %f' % self.recommender.embedding.get_value().max()
                    print '-1 item num is %d' % default_num

        return self

    def predict(self, user_id, pred_length):

        user_baskets = self.user_basket_dream[user_id]
        user_baskets_sizes = self.baskets_len[user_id]

        test_x = np.asarray(user_baskets[0:-1]).astype('int32')
        test_size = np.asarray(user_baskets_sizes[0:-1]).astype('int32')

        y_pred = self.recommender.predict(test_x, test_size)
        # print y_pred
        y_pred = [y for y in y_pred if y < self.n_item]
        y_pred.reverse()
        # print y_pred
        # print ''

        max_nbr_item = min(pred_length, len(y_pred))
        pred_basket = y_pred[:max_nbr_item]

        return pred_basket
