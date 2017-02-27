from tbp.tbp import *

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


def main():
    print 'Test TBP'

    pred_length = 5

    path = './dataset/'

    dataset = 'coop'  # tafeng
    coop_level = 'category'  # category, segment
    test_partition_type = 'random'  # fixed, random

    min_pred_length = 2
    max_pred_length = 21
    pred_lengths = range(min_pred_length, max_pred_length)

    print datetime.datetime.now(), 'Read dataset', dataset

    if dataset == 'tafeng':
        customers_data = read_data(path + 'tafeng.json')
        item2category = None
    elif dataset.startswith('coop'):
        customers_data = read_data(path + 'coop100.json')
        if coop_level == 'category':
            item2category = get_item2category(path + 'coop_categories_map.csv', category_index['categoria'])
        else:
            item2category = None
    else:
        print datetime.datetime.now(), 'Unknown dataset'
        return

    print datetime.datetime.now(), 'Customers', len(customers_data)

    if test_partition_type == 'fixed':
        split_mode = 'loo'
    elif test_partition_type == 'random':
        split_mode = 'rnd'
    else:
        print datetime.datetime.now(), 'Unkown test partition type'
        return

    print datetime.datetime.now(), 'Partition dataset into train / test'
    customers_train_set, customers_test_set = split_train_test(customers_data,
                                                               split_mode=split_mode,
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=2,
                                                               item2category=item2category)

    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print datetime.datetime.now(), 'Customers for test', len(customers_train_set), \
        '%.2f%%' % (100.0 * len(customers_train_set) / len(customers_data))

    print datetime.datetime.now(), 'Create and build models'
    customers_recsys = dict()
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set.keys():
        print datetime.datetime.now(), customer_id
        customer_train_set = customers_train_set[customer_id]
        tbp = TBP()
        tbp.build_model(customer_train_set)
        customers_recsys[customer_id] = tbp

    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        if customer_id not in customers_recsys:
            continue

        tbp = customers_recsys[customer_id]
        customer_data = customers_train_set[customer_id]['data']
        next_baskets = customers_test_set[customer_id]['data']

        for next_basket_id in next_baskets:
            day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')
            pred_basket = tbp.predict(customer_data, day_of_next_purchase, nbr_patterns=None, pred_length=pred_length)
            pred_basket = set([new2old[item] for item in pred_basket])

            next_basket = next_baskets[next_basket_id]['basket']
            next_basket = set(next_basket.keys())

            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)

    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Prediction performed in', end_time - start_time

    f1_values = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print datetime.datetime.now(), 'TBP', 'avg', stats['avg']


if __name__ == "__main__":
    main()
