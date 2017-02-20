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
