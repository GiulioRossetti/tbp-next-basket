import numpy as np

from sklearn.decomposition import NMF as SKNMF

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate

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
