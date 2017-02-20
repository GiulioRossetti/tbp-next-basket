import json
import random
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict


def read_data(filename):
    customers_data = dict()
    data = open(filename, 'r')
    for row in data:
        customer_data = json.loads(row)
        customer_id = customer_data['customer_id']
        customers_data[customer_id] = customer_data
    data.close()

    return customers_data


def write_data(filename, customers_data):
    newfile = open(filename, 'w')
    for customer_id in customers_data:
        customer_json_data = json.dumps(customers_data[customer_id])
        newfile.write(customer_json_data + '\n')
    newfile.flush()
    newfile.close()


def get_item2category(filename, category_level=7):
    df = pd.read_csv(filename, delimiter=';', skipinitialspace=True)
    item2category = dict()
    for row in df.values:
        cod_mkt_id = str(row[0])
        cod_mkt = row[1]
        item2category[cod_mkt_id] = cod_mkt[:category_level]
    return item2category


def get_date(basket_id, customer_data):
    basket_data = customer_data['data'][basket_id]
    date_str = '%s-%s-%s %s:%s:%s' % (
        basket_data['anno'], basket_data['mese_n'], basket_data['giorno_n'],
        basket_data['ora'], basket_data['minuto'], '0')
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return date


def split_train_test(customers_data, split_mode='loo', min_number_of_basket=10, min_basket_size=1,
                     max_basket_size=float('inf'), min_item_occurrences=1, item2category=None):
    """
    split_mode = 'loo' leave one out
    split_mode = 70 training percentage (70-30)
    min_number_of_basket = 10 minimum number of baskets per customer
    min_basket_size = 1 minimum basket size
    max_basket_size = float('inf') maximum basket size
    """

    customers_train_set = dict()
    customers_test_set = dict()

    for customer_id in customers_data:
        customer_data = customers_data[customer_id]

        if len(customer_data['data']) < min_number_of_basket:
            continue

        if item2category is not None:
            for basket_id in customer_data['data']:
                basket = customer_data['data'][basket_id]['basket']
                basket_category = dict()
                for item in basket:
                    category = item2category[item]
                    length = len(basket[item])
                    if category not in basket_category:
                        basket_category[category] = [0] * length
                    for i in xrange(length):
                        basket_category[category][i] += basket[item][i]
                customer_data['data'][basket_id]['basket'] = basket_category

        train_set = dict()
        test_set = dict()

        if split_mode == 'loo':
            split_index = len(customer_data['data']) - 1
        elif split_mode == 'rnd':
            max_test_nbr = int(np.round(len(customer_data['data']) * 0.1))
            if max_test_nbr < 2:
                test_nbr = 1
            else:
                test_nbr = random.randint(2, max_test_nbr)
            split_index = len(customer_data['data']) - test_nbr
        else:
            if int(split_mode) > 0:
                train_percentage = int(split_mode)
                split_index = int(len(customer_data['data']) * train_percentage / 100.0)
            else:
                test_nbr = -int(split_mode)
                split_index = len(customer_data['data']) - test_nbr

        sorted_basket_ids = sorted(customer_data['data'])
        train_basket_ids = sorted_basket_ids[:split_index]
        test_basket_ids = sorted_basket_ids[split_index:]

        if min_item_occurrences > 1:
            item_count = defaultdict(int)
            for basket_id in train_basket_ids:
                basket = customer_data['data'][basket_id]['basket'].keys()
                for item in basket:
                    item_count[item] += 1

            for basket_id in train_basket_ids:
                basket = customer_data['data'][basket_id]['basket'].keys()
                for item in basket:
                    if item_count[item] < min_item_occurrences:
                        del customer_data['data'][basket_id]['basket'][item]

        for basket_id in train_basket_ids:
            basket = customer_data['data'][basket_id]['basket'].keys()
            if min_basket_size - 1 < len(basket) < max_basket_size:
                train_set[basket_id] = customer_data['data'][basket_id]

        for basket_id in test_basket_ids:
            basket = customer_data['data'][basket_id]['basket'].keys()
            if min_basket_size - 1 < len(basket) < max_basket_size:
                test_set[basket_id] = customer_data['data'][basket_id]

        if len(train_set) == 0 or len(test_set) == 0:
            continue

        customers_train_set[customer_id] = {'customer_id': customer_id, 'data': train_set}
        customers_test_set[customer_id] = {'customer_id': customer_id, 'data': test_set}

    return customers_train_set, customers_test_set


def data2baskets(customer_data):
    baskets = list()
    for i, basket_id in enumerate(sorted(customer_data['data'])):
        basket_data = customer_data['data'][basket_id]['basket']
        basket = list()
        for item in basket_data:
            basket.append(item)
        baskets.append(basket)

    return baskets


def remap_items(baskets):
    new2old = dict()
    old2new = dict()
    new_baskets = list()
    for u, user_baskets in enumerate(baskets):
        new_user_baskets = list()
        for t, basket in enumerate(user_baskets):
            new_basket = list()
            for i in basket:
                if i not in old2new:
                    new_i = len(old2new)
                    old2new[i] = new_i
                    new2old[new_i] = i
                new_basket.append(old2new[i])
            new_user_baskets.append(new_basket)
        new_baskets.append(new_user_baskets)
    return new_baskets, new2old, old2new


def remap_items_with_data(baskets):
    new2old = dict()
    old2new = dict()
    new_baskets = dict()
    for customer_id in baskets:
        new_user_baskets = {'customer_id': customer_id, 'data': dict()}
        user_baskets = baskets[customer_id]
        for basket_id in user_baskets['data']:
            basket = user_baskets['data'][basket_id]['basket']
            new_basket = dict()
            for i in basket:
                if i not in old2new:
                    new_i = len(old2new)
                    old2new[i] = new_i
                    new2old[new_i] = i
                new_basket[old2new[i]] = basket[i]

            new_user_baskets['data'][basket_id] = dict()
            new_user_baskets['data'][basket_id]['basket'] = new_basket
        new_baskets[customer_id] = new_user_baskets

    return new_baskets, new2old, old2new


def get_items(baskets):
    items = dict()
    for user_baskets in baskets:
        for basket in user_baskets:
            for item in basket:
                items[item] = 0
    return items


def count_users_items(baskets):
    user_count = defaultdict(int)
    users_item_count = defaultdict(lambda: defaultdict(int))
    item_count = defaultdict(int)
    for u, user_basket in enumerate(baskets):

        user_item_count = defaultdict(int)
        num_purchases = 0
        for basket in user_basket:
            for item in basket:
                num_purchases += 1.0
                item_count[item] += 1.0
                user_item_count[item] += 1.0

        user_count[u] = num_purchases
        users_item_count[u] = user_item_count

    return user_count, item_count, users_item_count


category_index = {
    'settore': 2,
    'reparto': 4,
    'categoria': 7,
    'sottocategoria': 9,
    'segmento': 11,
}
