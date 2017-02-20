import math

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


def sigmoid(x):
    return math.exp(-np.logaddexp(0, -x))


def logistic(x, y):
    x_dot_y = np.dot(x, y)
    delta = sigmoid(x_dot_y)
    return delta


class HRM:

    def __init__(self, n_user, n_item, u_dim, v_dim, neg_samples=0,
                 n_epoch=20, alpha=0.01, lambda_r=0.001, decay=0.9, drop=0.5, n_thread=10, verbose=False):

        self.n_user = n_user
        self.n_item = n_item
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.neg_samples = neg_samples
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.lambda_r = lambda_r
        self.decay = decay
        self.drop = drop
        self.n_thread = n_thread
        self.verbose = verbose

        self.__state = 'initialized'

        self.map_user_item_set = None
        self.context_key_item_map = dict()

    def __get_context_maxpooling_and_droopout(self, basket, uid):

        context = np.zeros(self.v_dim)
        for item in basket:
            for d in xrange(self.v_dim):
                if self.V[item][d] > context[d]:
                    context[d] = self.V[item][d]
                    self.context_key_item_map[d] = '%s_i' % item

        # we randomly drop some items
        for d in xrange(self.v_dim):
            if random.random() > self.drop:
                context[d] = -10.0

        for d in xrange(self.u_dim):
            if self.U[uid][d] > context[d]:
                context[d] = self.U[uid][d]
                self.context_key_item_map[d] = '%s_u' % uid

        return context

    def __get_negative_items(self, uid):
        negative_items = list()
        items_bought = self.map_user_item_set[uid]
        total_step = 3000

        while len(negative_items) < self.neg_samples:
            total_step -= 1
            if total_step < 0:
                return negative_items
            neg_item = random.randint(0, self.n_item - 1)
            if neg_item not in items_bought:
                negative_items.append(neg_item)

        return negative_items

    def __get_negative_item_map(self, negative_items):
        negative_item_map = dict()
        for item in negative_items:
            item_neg = self.V[item, :]
            negative_item_map[item] = item_neg

        return negative_item_map

    def __get_neg_loss(self, negative_item_map, context):
        neg_loss_map = dict()
        for item in negative_item_map:
            f_neg = logistic(context, negative_item_map[item])
            neg_loss_map[item] = f_neg

        return neg_loss_map

    def __get_optimization_value(self, f, neg_loss_map):
        value = 0.0
        # for item in neg_loss_map:
        #     f_neg = math.log(1.0 - neg_loss_map[item])
        #     value += f_neg
        # value += math.log(f)

        for item in neg_loss_map:
            if 1.0 - neg_loss_map[item] > 0.0:
                f_neg = math.log(1.0 - neg_loss_map[item])
            else:
                f_neg = math.log(1.0 - 0.999999)
            value += f_neg

        value += math.log(f)
        return value

    def __update_rule(self, basket_with_context):
        uid = basket_with_context['uid']
        pitem = basket_with_context['pitem']
        basket = basket_with_context['basket']

        context = self.__get_context_maxpooling_and_droopout(basket, uid)
        item_predict = self.V[pitem, :]
        f = logistic(context, item_predict)

        delta_item_predict = context * (1.0 - f) * self.alpha
        new_item_predict = (delta_item_predict + item_predict) - item_predict * self.alpha * self.lambda_r

        # obtain negative samples fro uid
        negative_items = self.__get_negative_items(uid)
        negative_item_map = self.__get_negative_item_map(negative_items)
        neg_loss_map = self.__get_neg_loss(negative_item_map, context)
        value = self.__get_optimization_value(f, neg_loss_map)

        # update
        matrix_item_predict = item_predict   #np.asmatrix(item_predict)
        delta_context_positive = matrix_item_predict * self.alpha * (1.0 - f)
        # print delta_context_positive.shape

        neg_new_vec_map = dict()
        for item in negative_item_map:
            item_neg = negative_item_map[item]
            f_neg = neg_loss_map[item]
            delta_item_neg = (context * -self.alpha * f_neg) - item_neg * self.lambda_r * self.alpha
            new_item_neg = delta_item_neg + item_neg
            matrix_item_neg = item_neg    #np.matrix(item_neg)
            neg_new_vec_map[item] = new_item_neg
            delta_context_neg = matrix_item_neg * f_neg * self.alpha
            delta_context_positive -= delta_context_neg

        for d in xrange(len(new_item_predict)):
            self.V[pitem][d] = new_item_predict[d]

        for d in self.context_key_item_map:
            val = self.context_key_item_map[d]
            strs = val.split('_')
            if strs[1] == 'i':
                item = int(strs[0])
                # delta_context_positive[0][d]
                item_val = self.V[item][d] + delta_context_positive[d] - self.V[item][d] * self.lambda_r * self.alpha
                self.V[item][d] = item_val
            elif strs[1] == 'u':
                uid = int(strs[0])
                # delta_context_positive[0][d]
                user_val = self.U[uid][d] + delta_context_positive[d] - self.U[uid][d] * self.lambda_r * self.alpha
                self.U[uid][d] = user_val

        for item in neg_new_vec_map:
            v = neg_new_vec_map[item]
            for d in xrange(len(v)):
                self.V[item][d] = v[d]

        return value

    def __get_user_bought_item_set(self, baskets):
        if self.map_user_item_set is None:
            self.map_user_item_set = list()
            for user_baskets in baskets:
                item_set = defaultdict(int)
                for basket in user_baskets:
                    for item in basket:
                        item_set[item] += 1

                self.map_user_item_set.append(item_set)

        return self.map_user_item_set

    # def __get_test_map_tran(self, baskets):
    #     self.map_user_test_tran = list()
    #     for u, user_baskets in enumerate(baskets):
    #         self.map_user_test_tran.append(user_baskets[len(user_baskets) - 1])
    #     return self.map_user_test_tran

    def __get_user_tran_context(self, baskets):
        self.user_tran_context = list()
        for uid, user_baskets in enumerate(baskets):
            for bid in xrange(0, len(user_baskets)-1):
                basket = user_baskets[bid]

                bid_p1 = bid + 1
                for pitem in user_baskets[bid_p1]:
                    basket_with_context = {
                        'uid': uid,
                        'basket': basket,
                        'pitem': pitem
                    }
                    self.user_tran_context.append(basket_with_context)

        return self.user_tran_context

    def get_state(self):
        return self.__state

    def __init_matrices(self):
        self.U = (np.random.rand(self.n_user, self.u_dim) * 2.0 - 1.0) / self.u_dim
        self.V = (np.random.rand(self.n_item, self.v_dim) * 2.0 - 1.0) / self.v_dim

    def build_model(self, baskets):
        self.__state = 'built'
        self.__init_matrices()
        self.__get_user_bought_item_set(baskets)
        # self.__get_test_map_tran(baskets)
        self.__get_user_tran_context(baskets)
        # for a in self.user_tran_context:
        #     print a
        for i in xrange(self.n_epoch):
            random.shuffle(self.user_tran_context)
            value = 0.0
            # eventuale divisione per thread qui
            for basket_with_context in self.user_tran_context:
                rule_value = self.__update_rule(basket_with_context)
                value += rule_value
            self.alpha *= self.decay
            if self.verbose:
                print datetime.datetime.now(), 'Epoch %s, loss: %s' % (i, value)

        return self

    def update_model(self, new_baskets):
        self.build_model(new_baskets)
        return self

    def __get_all_max_pooling_and_dropout(self, user_id, last_basket):
        context = np.zeros(self.v_dim)
        for item in last_basket:
            item_vec = self.V[item, :]
            for d in xrange(self.v_dim):
                if item_vec[d] > context[d]:
                    context[d] = item_vec[d]
        for d in xrange(self.u_dim):
            if context[d] < self.U[user_id][d]:
                context[d] = self.U[user_id][d]
        return context

    def predict(self, user_id, last_basket, pred_length=5):
        if self.__state != 'built':
            raise Exception('Model not built, prediction not available')

        vec_context = self.__get_all_max_pooling_and_dropout(user_id, last_basket)
        scores = np.dot(self.V, vec_context)
        item_rank = {k: v for k, v in enumerate(scores)}
        # print item_rank

        max_nbr_item = min(pred_length, len(item_rank))
        pred_basket = sorted(item_rank, key=item_rank.get, reverse=True)[:max_nbr_item]

        return pred_basket
#
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
#
# n_user = len(baskets)
# n_item = len(items)
# u_dim = 5
# v_dim = 5
# neg_samples = 5
# n_epoch = 100
# alpha = 0.05
# lambda_r = 0.01
# decay = 0.8
# drop = 0.2
# n_thread = 10
#
#
# hrm = HRM(n_user, n_item, u_dim, v_dim, neg_samples=neg_samples,
#                  n_epoch=n_epoch, alpha=alpha, lambda_r=lambda_r, decay=decay, drop=drop)
# hrm.build_model(baskets)
#
# user_id = 0
#
# pred_basket = hrm.predict(user_id, baskets[user_id][len(baskets[user_id]) - 1])
#
# print pred_basket


def main():
    print 'HRM Competitor Test'

    pred_length = 5
    n_dim = 50
    u_dim = n_dim
    v_dim = n_dim
    neg_samples = 25
    n_epoch = 10
    alpha = 0.01
    lambda_r = 0.001
    decay = 0.9
    drop = 0.6

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
        # item2category = None
    else:
        print 'Unnown dataset'
        return

    print datetime.datetime.now(), 'Customers', len(customers_data)

    customers_train_set, customers_test_set = split_train_test(customers_data,
                                                               split_mode='loo',
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=1,
                                                               item2category=item2category)

    print datetime.datetime.now(), 'Customers for test', len(customers_train_set), \
        '%.2f%%' % (100.0*len(customers_train_set)/len(customers_data))

    print datetime.datetime.now(), 'Create and build models'
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

    n_user = len(customers_baskets)
    n_item = len(items)
    hrm = HRM(n_user, n_item, u_dim, v_dim, neg_samples=neg_samples,
                     n_epoch=n_epoch, alpha=alpha, lambda_r=lambda_r, decay=decay, drop=drop, verbose=True)
    hrm.build_model(customers_baskets)
    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)

    # print 'n items', len(items)

    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        user_id = customerid_userid[customer_id]
        last_basket = customers_baskets[user_id][len(customers_baskets[user_id]) - 1]
        pred_basket = hrm.predict(user_id, last_basket, pred_length)
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
    print datetime.datetime.now(), 'HRM', 'avg', stats['avg']
    print datetime.datetime.now(), 'HRM', 'iqm', stats['iqm']

    # for k, v in stats.iteritems():
    #     print k, v


if __name__ == "__main__":
    main()
