import math
import numpy as np

import datetime

def freedman_diaconis(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    n = len(x)
    h = 2.0 * iqr / n**(1.0/3.0)
    k = math.ceil((np.max(x) - np.min(x))/h)
    return k


def struges(x):
    n = len(x)
    k = math.ceil( np.log2(n) ) + 1
    return k


def estimate_nbr_bins(x):
    if len(x) == 1:
        return 1
    k_fd = freedman_diaconis(x) if len(x) > 2 else 1
    k_struges = struges(x)
    if k_fd == float('inf') or np.isnan(k_fd):
        k_fd = np.sqrt(len(x))
    k = max(k_fd, k_struges)
    return k


def estimate_basket_length(baskets):
    basket_lengths = list()
    basket_ids = baskets['data']

    for basket_id in basket_ids:
        basket = baskets['data'][basket_id]['basket']

        basket_len = len(basket)
        basket_lengths.append(basket_len)

    if len(basket_lengths) <= 10:
        return int(np.round(np.median(basket_lengths)))

    nbr_bins = np.round(estimate_nbr_bins(basket_lengths))
    val, bins = np.histogram(basket_lengths, bins=nbr_bins)
    ebl = int(np.round(bins[np.argmax(val)]))
    ebl = ebl + 1 if ebl == 1 else ebl

    return ebl


def estimate_month_basket_length(baskets):
    month_basket_lenght = [[] for x in xrange(12)]

    basket_ids = baskets['data']

    for basket_id in basket_ids:
        date_object = datetime.datetime.strptime(basket_id[0:10], '%Y_%m_%d')
        basket = baskets['data'][basket_id]['basket']
        month_id = date_object.month - 1

        basket_len = len(basket)
        month_basket_lenght[month_id].append(basket_len)

    month_ebl = list()
    for month_id in xrange(12):
        nbr_bins = estimate_nbr_bins(month_basket_lenght[month_id])
        nbr_bins = np.round(nbr_bins)
        val, bins = np.histogram(month_basket_lenght[month_id], bins=nbr_bins)
        mebl = int(np.round(bins[np.argmax(val)]))
        mebl = mebl + 1 if mebl == 1 else mebl
        month_ebl.append(mebl)

    return month_ebl

