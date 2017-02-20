import re
import math
import datetime
import itertools
import numpy as np

from collections import defaultdict

from utils.data_management import *
from evaluation.evaluation_measures import *
from evaluation.calculate_aggregate_statistics import calculate_aggregate


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


def logistic(x, b=10.0, m=0.5):
    return 1.0/(1.0 + math.exp(-b*(x-m)))


class TARSNode(object):
    """
    A node in the RP tree.
    """

    def __init__(self, value, count, parent, timestamp):
        """
        Create the node.
        """
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = list()
        self.timeseries = list()
        if timestamp is not None:
            if type(timestamp) is list:
                self.timeseries.extend(timestamp)
            else:
                self.timeseries.append(timestamp)

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value, timestamp):
        """
        Add a node as a child node.
        """
        child = TARSNode(value, 1, self, timestamp)
        self.children.append(child)
        return child

    def disp(self, ind=1):
        if self.timeseries is not None:
            timestamps = ''
            for d in self.timeseries:
                if d is not None:
                    timestamps += '(%s-%s),' % (d[0].day, d[1].day)
        else:
            timestamps = None
        print '  ' * ind, '%s:%s %s' % (self.value, self.count, timestamps)
        for child in self.children:
            child.disp(ind + 1)


class TARSTree(object):
    """
    A recurring frequent pattern tree.
    """

    def __init__(self, baskets, root_value,
                 root_count,
                 root_timeseries,
                 item_period_thr_parent=None,
                 item_min_period_support_parent=None,
                 item_min_expected_espp_parent=None,
                 min_nbr_iat_parent=None):
        """
        Initialize the tree.
        """

        self.item_period_thr_parent = item_period_thr_parent
        self.item_min_period_support_parent = item_min_period_support_parent
        self.item_min_expected_espp_parent = item_min_expected_espp_parent
        self.min_nbr_iat_parent = min_nbr_iat_parent

        self.item_support = None
        self.item_periods = None
        self.item_timeseries = None
        self.single_item_timeseries = None
        self.single_item_support = None
        self.single_item_support_stable = None
        self.item_period_support = None
        self.item_period_days_passed = None
        self.item_period_thr = None
        self.item_estimated_nbr_periods = None
        self.item_estimated_min_period_support = None
        self.estimated_nbr_periods_item_period_support = None
        self.item_min_period_support = None
        self.item_tot_support_interesting_periods = None
        self.item_nbr_periods = None
        self.nbr_periods_espp_list = None
        self.item_tot_interesting_support_over_nbr_periods = None
        self.headers = None
        self.item_min_expected_espp = None
        self.min_nbr_iat = None

        self.preprocessing(baskets)

        self.root = None

        self.build_fptree(baskets,
                          root_value,
                          root_count,
                          root_timeseries)

    def preprocessing(self, baskets):
        """
        Create a dictionary of items with occurrences above the threshold and build the header table.
        """

        # Calculates support, inter arrival times, and build the timeseries
        self.item_support = dict()
        self.item_periods = dict()
        self.item_timeseries = dict()

        self.single_item_timeseries = dict()
        self.single_item_support = dict()
        self.single_item_support_stable = dict()

        sorted_basket_ids = sorted(baskets['data'])

        start_time = datetime.datetime.now()
        for t0, basket_id0 in enumerate(sorted_basket_ids):
            date_object0 = datetime.datetime.strptime(basket_id0[0:10], '%Y_%m_%d')
            basket0 = baskets['data'][basket_id0]['basket']

            if t0 < len(sorted_basket_ids) - 1:
                t1 = t0 + 1
                basket_id1 = sorted_basket_ids[t1]
                date_object1 = datetime.datetime.strptime(basket_id1[0:10], '%Y_%m_%d')
                basket1 = baskets['data'][basket_id1]['basket']

                sorted_b0 = sorted(basket0)
                sorted_b1 = sorted(basket1)
                sorted_others = sorted(self.single_item_timeseries)

                # calcualte periods, timeseries and support with basket t0 and t1
                for item0 in sorted_b0:
                    for item1 in sorted_b1:

                        item = tuple([tuple([item0]), tuple([item1])])

                        # sol funzionante
                        if item not in self.item_support:
                            self.item_support[item] = 1.0
                            self.item_timeseries[item] = [(date_object0, date_object1)]
                            days_from_last_bought = 1.0 * (date_object1 - date_object0).days
                            self.item_periods[item] = [days_from_last_bought]
                        else:
                            self.item_support[item] += 1.0
                            self.item_timeseries[item].append((date_object0, date_object1))
                            days_from_last_bought = 1.0 * (date_object1 - date_object0).days
                            self.item_periods[item].append(days_from_last_bought)

                        # sol teoricamente piu corretta
                        # if item not in self.item_support:
                        #     self.item_support[item] = 1.0
                        #     self.item_timeseries[item] = [(date_object0, date_object1)]
                        #     # days_from_last_bought = 1.0 * (date_object1 - date_object0).days
                        #     self.item_periods[item] = []
                        # else:
                        #     self.item_support[item] += 1.0
                        #     self.item_timeseries[item].append((date_object0, date_object1))
                        #     days_from_last_bought = 1.0 * (date_object0 - self.item_timeseries[item][-1][1]).days
                        #     self.item_periods[item].append(days_from_last_bought)

                    if item0 not in self.single_item_timeseries:
                        self.single_item_timeseries[item0] = [date_object0]
                        self.single_item_support[item0] = 1.0
                        self.single_item_support_stable[item0] = 1.0
                    else:
                        self.single_item_timeseries[item0].append(date_object0)
                        self.single_item_support[item0] += 1.0
                        self.single_item_support_stable[item0] += 1.0

                # calcualte periods, timeseries and support with past baskets and t1
                for item0 in sorted_others:
                    for item1 in sorted_b1:

                        if item0 in basket0 or \
                                (item1 in self.single_item_timeseries and
                                         self.single_item_timeseries[item0][-1] < self.single_item_timeseries[item1][-1]):
                            continue

                        item = tuple([tuple([item0]), tuple([item1])])

                        date_last_purchase = self.single_item_timeseries[item0][-1]

                        # sol funzionante
                        if item not in self.item_support:
                            self.item_support[item] = 1.0
                            self.item_timeseries[item] = [(date_last_purchase, date_object1)]
                            days_from_last_bought = 1.0 * (date_object1 - date_last_purchase).days
                            self.item_periods[item] = [days_from_last_bought]
                        else:
                            self.item_support[item] += 1.0
                            self.item_timeseries[item].append((date_last_purchase, date_object1))
                            days_from_last_bought = 1.0 * (date_object1 - date_last_purchase).days
                            self.item_periods[item].append(days_from_last_bought)

                        # sol teoricamente piu corretta
                        # if item not in self.item_support:
                        #     self.item_support[item] = 1.0
                        #     self.item_timeseries[item] = [(date_last_purchase, date_object1)]
                        #     # days_from_last_bought = 1.0 * (date_object1 - date_last_purchase).days
                        #     self.item_periods[item] = []
                        # else:
                        #     self.item_support[item] += 1.0
                        #     self.item_timeseries[item].append((date_last_purchase, date_object1))
                        #     days_from_last_bought = 1.0 * (date_object0 - self.item_timeseries[item][-1][1]).days
                        #     self.item_periods[item].append(days_from_last_bought)

        end_time = datetime.datetime.now()
        # print 'prima inizzializzazione', end_time - start_time

        start_time = datetime.datetime.now()
        if self.min_nbr_iat_parent is None:
            nbr_iat_list = list()
            for item in self.item_periods:
                nbr_iat_list.append(len(self.item_periods[item]))
            self.min_nbr_iat = np.floor(np.mean(nbr_iat_list))
            # self.min_nbr_iat = np.percentile(nbr_iat_list, 50)
        else:
            self.min_nbr_iat = self.min_nbr_iat_parent
        end_time = datetime.datetime.now()
        # print 'stima min nbr iat', end_time - start_time

        start_time = datetime.datetime.now()
        # Calculates the support for each period
        self.item_period_support = dict()
        self.item_period_days_passed = dict()
        if self.item_period_thr_parent is None:
            self.item_period_thr = dict()

        for item in self.item_periods.keys():

            if self.item_period_thr_parent is not None and item not in self.item_period_thr_parent:
                continue

            self.item_period_support[item] = [0.0]
            self.item_period_days_passed[item] = [0.0]

            inter_arrival_times = self.item_periods[item]

            if self.item_period_thr_parent is None:
                if len(inter_arrival_times) < self.min_nbr_iat:
                    del self.item_support[item]
                    del self.item_periods[item]
                    del self.item_period_support[item]
                    del self.item_period_days_passed[item]
                    continue
                else:
                    per_thr = np.round(np.mean(inter_arrival_times))
                    # perThr = np.round(np.median(self.item_periods[item]))
                    # perThr = np.round(np.percentile(self.item_periods[item], 25))
                    self.item_period_thr[item] = per_thr
            else:
                per_thr = self.item_period_thr_parent[item]

            for iat in inter_arrival_times:
                if iat <= per_thr:
                    self.item_period_support[item][-1] += 1.0
                    self.item_period_days_passed[item][-1] += iat
                else:
                    self.item_period_support[item].append(0.0)
                    self.item_period_days_passed[item].append(0.0)
        end_time = datetime.datetime.now()
        # print 'divide in periodi e conta supporti', end_time - start_time

        start_time = datetime.datetime.now()
        # Estimatets the number of periods with respect to the estimated min period support
        if self.item_min_period_support_parent is None:

            self.item_estimated_nbr_periods = dict()
            self.item_estimated_min_period_support = dict()

            for item in self.item_period_support:
                self.item_estimated_nbr_periods[item] = 0.0
                self.item_estimated_min_period_support[item] = np.mean(self.item_period_support[item])
                for period_support in self.item_period_support[item]:
                    if period_support >= self.item_estimated_min_period_support[item]:
                        self.item_estimated_nbr_periods[item] += 1.0

            # Group items with respect to the estimated nubmer of periods
            nbr_bins = np.round(estimate_nbr_bins(self.item_estimated_nbr_periods.values()))
            _, bins = np.histogram(self.item_estimated_nbr_periods.values(), bins=nbr_bins)
            for item in list(self.item_estimated_nbr_periods.keys()):
                index = 0
                while bins[index] < self.item_estimated_nbr_periods[item]:
                    index += 1
                self.item_estimated_nbr_periods[item] = np.round(bins[index])

            # Group period support with respect to the estimated number of periods
            self.estimated_nbr_periods_item_period_support = defaultdict(list)
            for item in self.item_period_support:
                estimated_nbr_periods = self.item_estimated_nbr_periods[item]
                for period_support in self.item_period_support[item]:
                    if period_support >= self.item_estimated_min_period_support[item]:
                        self.estimated_nbr_periods_item_period_support[estimated_nbr_periods].append(period_support)

            # Calcualtes minimum item support with respect to the period support grouping
            self.item_min_period_support = dict()
            for item in self.item_period_support.keys():
                estimated_nbr_periods = self.item_estimated_nbr_periods[item]
                min_period_support = np.percentile(
                    self.estimated_nbr_periods_item_period_support[estimated_nbr_periods], 25)
                self.item_min_period_support[item] = min_period_support
        end_time = datetime.datetime.now()
        # print 'stima periodi interessanti e stima min period sup', end_time - start_time

        start_time = datetime.datetime.now()
        # Calcualtes the number of periods with respect to the minimum support and to the minimum item support
        # Moreover calculates the total support in interesting periods
        self.item_tot_support_interesting_periods = defaultdict(int)
        self.item_nbr_periods = defaultdict(float)
        for item in self.item_period_support:

            if self.item_min_period_support_parent is None:
                min_ps = self.item_min_period_support[item]
            else:
                min_ps = self.item_min_period_support_parent[item]

            for period_support in self.item_period_support[item]:
                if period_support >= min_ps:
                    self.item_nbr_periods[item] += 1.0
                    self.item_tot_support_interesting_periods[item] += period_support
        end_time = datetime.datetime.now()
        # print 'conta periodi interessanti', end_time - start_time

        start_time = datetime.datetime.now()
        # Correct nubmer of periods according to similar items
        if self.item_min_expected_espp_parent is None:

            nbr_bins = np.round(estimate_nbr_bins(self.item_nbr_periods.values()))
            _, bins = np.histogram(self.item_nbr_periods.values(), bins=nbr_bins)
            for item in list(self.item_nbr_periods.keys()):
                index = 0
                while bins[index] < self.item_nbr_periods[item]:
                    index += 1
                self.item_nbr_periods[item] = np.round(bins[index])
        end_time = datetime.datetime.now()
        # print 'corregge nuemro periodi interessanti', end_time - start_time

        start_time = datetime.datetime.now()
        # Calculates the total support on interesting periods on number of periods
        # i.e. the expected support per period
        self.nbr_periods_espp_list = defaultdict(list)
        self.item_tot_interesting_support_over_nbr_periods = dict()

        for item in self.item_period_support:
            nbr_periods = self.item_nbr_periods[item]
            if nbr_periods > 0:
                espp = self.item_tot_support_interesting_periods[item] / nbr_periods
                self.item_tot_interesting_support_over_nbr_periods[item] = espp
                self.nbr_periods_espp_list[nbr_periods].append(espp)
            else:
                self.item_tot_interesting_support_over_nbr_periods[item] = 0.0
                self.nbr_periods_espp_list[nbr_periods].append(0.0)
        end_time = datetime.datetime.now()
        # print 'valuta ratio totale supporto su periodi interessanti su totale periodi', end_time - start_time

        start_time = datetime.datetime.now()
        # Prepare header and clean item support
        self.headers = dict()
        if self.item_min_expected_espp_parent is None:
            self.item_min_expected_espp = dict()

        # TODO calcolare quanti pattern a item singoli vengono abbattuti
        item_appearing_as_head = dict()
        item_appearing_as_tail = dict()
        for item in list(self.item_support.keys()):

            if self.item_min_expected_espp_parent is not None and item not in self.item_min_expected_espp_parent:
                continue

            espp = self.item_tot_interesting_support_over_nbr_periods[item]
            if self.item_min_expected_espp_parent is None:
                min_espp = np.percentile(self.nbr_periods_espp_list[self.item_nbr_periods[item]], 25)
                self.item_min_expected_espp[item] = min_espp
            else:
                min_espp = self.item_min_expected_espp_parent[item]
            if espp < min_espp or not self.item_nbr_periods[item]:
                del self.item_support[item]
            else:
                self.headers[item] = None
                item_appearing_as_head[item[0][0]] = 0
                item_appearing_as_tail[item[1][0]] = 0
        end_time = datetime.datetime.now()
        # print 'definisce min expected support per period', end_time - start_time

        start_time = datetime.datetime.now()
        for item in list(self.single_item_timeseries.keys()):
            if item not in item_appearing_as_head and item not in item_appearing_as_tail:
                del self.single_item_timeseries[item]
                del self.single_item_support[item]
        end_time = datetime.datetime.now()
        # print 'rimuove single item', end_time - start_time

    def build_fptree(self, baskets, root_value, root_count, root_timeseries):
        """
        Build the FP tree and return the root node.
        """

        self.root = TARSNode(root_value, root_count, None, root_timeseries)

        single_item_last_purchase = dict()

        sorted_basket_ids = sorted(baskets['data'])

        for t0, basket_id0 in enumerate(sorted_basket_ids):
            date_object0 = datetime.datetime.strptime(basket_id0[0:10], '%Y_%m_%d')
            basket0 = baskets['data'][basket_id0]['basket']

            if t0 < len(sorted_basket_ids) - 1:

                t1 = t0 + 1
                basket_id1 = sorted_basket_ids[t1]
                date_object1 = datetime.datetime.strptime(basket_id1[0:10], '%Y_%m_%d')
                basket1 = baskets['data'][basket_id1]['basket']

                sorted_b0 = sorted(basket0)
                sorted_b1 = sorted(basket1)
                sorted_others = sorted(single_item_last_purchase)

                # build tree with past basket at t0 and t1
                items = list()
                intervals = dict()
                for item0 in sorted_b0:
                    for item1 in sorted_b1:
                        item = tuple([tuple([item0]), tuple([item1])])
                        items.append(item)
                        intervals[item] = (date_object0, date_object1)

                    single_item_last_purchase[item0] = date_object0

                # build tree with past baskets and t1
                for item0 in sorted_others:
                    for item1 in sorted_b1:

                        if item0 in basket0 or \
                                (item1 in single_item_last_purchase and
                                             single_item_last_purchase[item0] < single_item_last_purchase[item1]):
                            continue

                        item = tuple([tuple([item0]), tuple([item1])])
                        items.append(item)
                        intervals[item] = (single_item_last_purchase[item0], date_object1)

                sorted_items = {item: self.item_support[item] for item in items if item in self.item_support}
                sorted_items = [i[0] for i in sorted(sorted_items.iteritems(),
                                                     key=lambda (k, v): (v, -k[0][0], -k[1][0]),
                                                     reverse=True)]
                sorted_intervals = [intervals[item] for item in sorted_items]

                child = self.root
                for item, intervals in zip(sorted_items, sorted_intervals):
                    child = self.insert_tree(item, child, intervals)

        return self

    def insert_tree(self, items, node, date_object_list):
        """
        Recursively grow FP tree.
        """

        first = items
        date_object = date_object_list

        child = node.get_child(first)
        if child is not None:
            child.count += 1
            child.timeseries.append(date_object)
        else:
            child = node.add_child(first, date_object)

            # Link it to header structure.
            if first in self.headers:
                if self.headers[first] is None:
                    self.headers[first] = child
                else:
                    current = self.headers[first]
                    while current.link is not None:
                        current = current.link
                    current.link = child

        return child

    ########################### pattern mining ###############################
    def tree_has_single_path(self, node, rec_dept=0):
        """
        If there is a single path in the tree, return True, else return False.
        """
        if rec_dept >= 950:
            return True
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0], rec_dept=rec_dept+1)

    def mine_patterns(self, rec_dept=0, max_rec_dept=1, patterns_subset=None, nbr_patterns=None,
                       get_items_in_order_of_occurrences=True):
        """
        Mine the constructed FP tree for frequent patterns.
        """
        if self.tree_has_single_path(self.root):
            return self.generate_pattern_list()
        else:
            patterns = self.mine_sub_trees(rec_dept, max_rec_dept, patterns_subset, nbr_patterns,
                                           get_items_in_order_of_occurrences)
            return self.zip_patterns(patterns)

    def zip_patterns(self, patterns):
        """
        Append suffix to patterns in dictionary if we are in a conditional FP tree.
        """
        suffix = self.root.value

        if suffix is not None:
            # We are in a conditional tree.
            new_patterns = {}
            for key in patterns.keys():
                item0 = [x for x in suffix[0]]
                item0.extend([x for x in key[0]])
                item0 = tuple(sorted(item0)) if len(key[0]) > 0 else suffix[0]

                item1 = [x for x in suffix[1]]
                item1.extend([x for x in key[1]])
                item1 = tuple(sorted(item1)) if len(key[1]) > 0 else suffix[1]

                item = tuple([item0, item1])
                new_patterns[item] = patterns[key]

            return new_patterns

        return patterns

    def generate_pattern_list(self):
        """
        Generate a list of patterns with support counts.
        """

        patterns = {}
        items = self.single_item_timeseries.keys()

        # If we are in a conditional tree, the suffix is a pattern on its own.
        if self.root.value is None:
            suffix_value = []
        else:
            suffix_value = self.root.value
            patterns[suffix_value] = [self.root.count, self.root.timeseries]

        for i in range(0, 2):
            for j in range(0, 2):
                if i == 0 and j == 0:
                    continue
                set_head = set(items) - set(suffix_value[0] if len(suffix_value) > 0 else [])
                for subset_head in itertools.combinations(set_head, min(i, len(set_head))):
                    set_tail = set(items) - set(suffix_value[1] if len(suffix_value) > 0 else [])
                    for subset_tail in itertools.combinations(set_tail, min(j, len(set_tail))):
                        # print suffix_value, subset_head, subset_tail
                        if len(suffix_value) > 0:
                            pattern = tuple([tuple(sorted(list(subset_head) + list(suffix_value[0]))),
                                             tuple(sorted(list(subset_tail) + list(suffix_value[1])))
                                             ])

                            sup_list_head = list()
                            for x in subset_head:
                                key = (tuple([x]), suffix_value[1])
                                if key in self.item_support:
                                    sup_list_head.append(self.item_support[key])

                            sup_list_tail = list()
                            for x in subset_tail:
                                key = (suffix_value[0], tuple([x]))
                                if key in self.item_support:
                                    sup_list_tail.append(self.item_support[key])

                            min_sup_list_head = float('inf') if len(sup_list_head) == 0 else min(sup_list_head)
                            min_sup_list_tail = float('inf') if len(sup_list_tail) == 0 else min(sup_list_tail)

                            if min_sup_list_head < float('inf') or min_sup_list_tail < float('inf'):
                                patterns[pattern] = [min(min_sup_list_head, min_sup_list_tail), self.root.timeseries]

        return patterns

    def mine_sub_trees(self, rec_dept, max_rec_dept, patterns_subset=None, nbr_patterns=None,
                       get_items_in_order_of_occurrences=True):
        """
        Generate subtrees and mine them for patterns.
        """

        patterns = dict()

        if rec_dept < max_rec_dept:

            # Get items in tree in reverse order of occurrences.
            # mining_order = [i[0] for i in sorted(self.item_support.iteritems(),
            #                                      key=lambda (k, v): (v, -k[0][0], -k[1][0]))]

            # for k,v in calculate_aggregate(self.item_support.values()).iteritems():
            #     print k, v

            mining_order = [i[0] for i in sorted(self.item_support.iteritems(),
                                                 key=lambda (k, v): (v, -k[0][0], -k[1][0]),
                                                 reverse=not get_items_in_order_of_occurrences)]
            if patterns_subset is not None:

                subset_item_support = dict()
                for item in mining_order:
                    if item in patterns_subset:
                        subset_item_support[item] = self.item_support[item]

                if nbr_patterns is not None:
                    nbr_patterns = min(nbr_patterns, len(subset_item_support))
                    mining_order_subset = sorted(subset_item_support, key=subset_item_support.get, reverse=True)[:nbr_patterns]
                    mining_order = mining_order_subset
                else:
                    mining_order_subset = list()
                    if len(subset_item_support) > 3:

                        min_sup_interesting = np.percentile(subset_item_support.values(), 90)
                        for item in mining_order:
                            if item in subset_item_support and subset_item_support[item] > min_sup_interesting:
                                mining_order_subset.append(item)

                        mining_order = mining_order_subset
                    else:
                        mining_order = list()

            elif nbr_patterns is not None:
                mining_order = mining_order[:nbr_patterns]

            for item in mining_order:

                if item not in self.headers:
                    continue

                if len(self.item_timeseries[item]) > 1:

                    espp = self.item_tot_interesting_support_over_nbr_periods[item]

                    if self.item_min_expected_espp_parent is None:
                        min_espp = self.item_min_expected_espp[item]
                    else:
                        min_espp = self.item_min_expected_espp_parent[item]

                    if espp >= -min_espp:

                        suffixes = list()
                        conditional_tree_input = {'customer_id': 0, 'data': dict()}
                        node = self.headers[item]

                        # Follow node links to get a list of all occurrences of a certain item.
                        while node is not None:
                            suffixes.append(node)
                            node = node.link

                        # For each occurrence of the item, trace the path back to the root node.
                        for suffix in suffixes:
                            parent = suffix.parent

                            while parent.parent is not None:

                                for i in xrange(parent.count):
                                    basket_id0 = re.sub("[-: ]", '_', str(parent.timeseries[i][0]))
                                    basket_id1 = re.sub("[-: ]", '_', str(parent.timeseries[i][1]))

                                    if basket_id0 not in conditional_tree_input['data']:
                                        conditional_tree_input['data'][basket_id0] = {'basket': dict()}

                                    if basket_id1 not in conditional_tree_input['data']:
                                        conditional_tree_input['data'][basket_id1] = {'basket': dict()}

                                    conditional_tree_input['data'][basket_id0]['basket'][parent.value[0][0]] = [1.0]
                                    conditional_tree_input['data'][basket_id1]['basket'][parent.value[1][0]] = [1.0]

                                parent = parent.parent

                        # Now we have the input for a subtree, so construct it and grab the patterns.
                        # Controlla se sono alla prima ricerca oppure no
                        if self.item_period_thr_parent is None:
                            item_period_thr_parent = self.item_period_thr
                            item_min_period_support_parent = self.item_min_period_support
                            item_min_expected_espp_parent = self.item_min_expected_espp
                            min_nbr_iat_parent = self.min_nbr_iat
                        else:
                            item_period_thr_parent = self.item_period_thr_parent
                            item_min_period_support_parent = self.item_min_period_support_parent
                            item_min_expected_espp_parent = self.item_min_expected_espp_parent
                            min_nbr_iat_parent = self.min_nbr_iat_parent

                        # Build the subtree
                        subtree = TARSTree(conditional_tree_input,
                                           root_value=item,
                                           root_count=self.item_support[item],
                                           root_timeseries=self.item_timeseries[item],
                                           item_period_thr_parent=item_period_thr_parent,
                                           item_min_period_support_parent=item_min_period_support_parent,
                                           item_min_expected_espp_parent=item_min_expected_espp_parent,
                                           min_nbr_iat_parent=min_nbr_iat_parent)

                        if len(subtree.root.children) > 0:

                            subtree_patterns = subtree.mine_patterns(rec_dept=rec_dept + 1,
                                                                     max_rec_dept=max_rec_dept,
                                                                     patterns_subset=patterns_subset)

                            # Insert subtree patterns into main patterns dictionary.
                            for pattern in subtree_patterns.keys():
                                new_pattern = pattern
                                if new_pattern in patterns:
                                    patterns[new_pattern][0] += subtree_patterns[pattern][0]
                                    patterns[new_pattern][1] += subtree_patterns[pattern][1]
                                else:
                                    patterns[new_pattern] = subtree_patterns[pattern]

        for item in self.item_support:
            new_pattern = item
            if new_pattern not in patterns:

                if self.item_min_expected_espp_parent is None:
                    min_espp = self.item_min_expected_espp[item]
                else:
                    if item not in self.item_min_expected_espp_parent:
                        continue

                    min_espp = self.item_min_expected_espp_parent[item]

                if self.item_tot_interesting_support_over_nbr_periods[item] >= min_espp:
                    patterns[new_pattern] = [self.item_support[item], self.item_timeseries[item]]

        return patterns


def timeseries2periods(rp, timeseries, tree):
    periods = dict()
    lower_bound_time = timeseries[0][0]
    upper_bound_time = timeseries[0][1]
    period_support = 0.0
    period_list = list()

    item_min = None
    item_sup_min = float('inf')
    for item0 in rp[0]:
        for item1 in rp[1]:
            item = tuple([(item0,), (item1,)])
            if item in tree.item_support and tree.item_support[item] < item_sup_min:
                item_sup_min = tree.item_support[item]
                item_min = item

    per_thr = tree.item_period_thr[item_min]
    min_ps = tree.item_min_period_support[item_min]

    for time_couple in timeseries:
        # print time_couple
        delta_t = (time_couple[1] - time_couple[0]).days

        if delta_t <= per_thr:
            if len(period_list) == 0:
                lower_bound_time = time_couple[0]
            period_support += 1.0
            period_list.append(delta_t)
        else:
            if period_support >= min_ps and len(period_list) > 0:
                periods[tuple([lower_bound_time, upper_bound_time])] = [period_list, period_support]
            period_support = 0.0
            period_list = list()

        upper_bound_time = time_couple[1]

    if period_support >= min_ps and len(period_list) > 0:
        periods[tuple([lower_bound_time, upper_bound_time])] = [period_list, period_support]

    return periods


def calculate_intervals_support(freq_rec_patterns, tree):
    rp_intervals_support = dict()
    max_rp_len = -float('inf')
    for rp, info in freq_rec_patterns.iteritems():
        periods = timeseries2periods(rp, info[1], tree)
        if len(periods) == 0:
            continue
        min_days = float('inf')
        max_days = -float('inf')
        tot_sup = 0.0

        for daterange in periods:
            days_passed = periods[daterange][0]
            period_support = periods[daterange][1]
            min_days = min(min_days, np.min(days_passed))
            max_days = max(max_days, np.max(days_passed))
            tot_sup += period_support
        avg_sup = tot_sup / len(periods)
        rp_intervals_support[rp] = [min_days, max_days, avg_sup, tot_sup]
        max_rp_len = max(max_rp_len, len(rp))

    return rp_intervals_support


def calcualte_active_rp(customer_data, rp_intervals_support, day_of_next_purchase, max_rp_len=3):
    sorted_basket_ids = sorted(customer_data, reverse=True)
    rp_purchases = defaultdict(int)
    rp_day_of_last_purchase = dict()
    nbr_rp_inactive = 0

    for t0, basket_id0 in enumerate(sorted_basket_ids):
        date_object0 = datetime.datetime.strptime(basket_id0[0:10], '%Y_%m_%d')
        basket0 = customer_data[basket_id0]['basket']

        if t0 < len(sorted_basket_ids) - 1:
            t1 = t0 + 1
            basket_id1 = sorted_basket_ids[t1]
            date_object1 = datetime.datetime.strptime(basket_id1[0:10], '%Y_%m_%d')
            basket1 = customer_data[basket_id1]['basket']

            # print nbr_rp_inactive, len(sorted_basket_ids)
            if nbr_rp_inactive >= len(rp_intervals_support):
                break

            for i in xrange(1, max_rp_len):
                for subset_tail in itertools.combinations(basket0, i):
                    for j in xrange(1, max_rp_len):
                        for subset_head in itertools.combinations(basket1, j):
                            rp = (subset_head, subset_tail)
                            # print rp
                            if rp in rp_intervals_support:
                                # print rp_intervals_support[rp]

                                interval_l = rp_intervals_support[rp][0]
                                interval_r = rp_intervals_support[rp][1]
                                support = rp_intervals_support[rp][2]

                                if (day_of_next_purchase - date_object0).days > np.round(interval_r * support):
                                    nbr_rp_inactive += 1
                                    # print 'AAAA'
                                    # print subset, interval_r * support, (day_of_next_purchase - date_object0).days
                                    continue

                                if rp not in rp_day_of_last_purchase:
                                    days_from_last_purchase = (day_of_next_purchase - date_object0).days
                                    rp_day_of_last_purchase[rp] = date_object0
                                else:
                                    days_from_last_purchase = (rp_day_of_last_purchase[rp] - date_object0).days
                                    rp_day_of_last_purchase[rp] = date_object0

                                if interval_l <= days_from_last_purchase <= interval_r:
                                    rp_purchases[rp] += 1
                                    if rp_purchases[rp] > support:
                                        rp_purchases[rp] = support

    return rp_purchases, rp_day_of_last_purchase


def calcualte_item_score(tree, rp_purchases, rp_intervals_support):
    item_score = defaultdict(int)
    for rp in rp_purchases:
        for item_tuple in rp:
            item = item_tuple[0]
            delay_sup = rp_intervals_support[rp][2] - rp_purchases[rp]
            item_score[item] += delay_sup

    for item in item_score:
        item_score[item] = tree.single_item_support.get(item, 0.0) + item_score.get(item, 0.0)

    # for item in item_score:
    #     v1 = 0.0 if tree.single_item_support[item] == 0.0 else 1.0/tree.single_item_support[item]
    #     v2 = 0.0 if item_score[item] == 0.0 else 1.0/item_score[item]
    #     item_score[item] = 2.0 / (v1 + v2)

    # for item in item_score:
    #     item_score[item] = (tree.single_item_support[item] + item_score[item]) / 2.0

    if len(item_score) == 0:
        item_score = tree.single_item_support_stable

    return dict(item_score)


# customer_data = {
#   "customer_id": 0,
#   "data": {
#     "2000_01_01_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 1,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "g": [1], "h": [1]},
#         },
#     "2000_01_02_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 4,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "c": [1], "d": [1]},
#         },
#     "2000_01_03_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 8,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "e": [1], "f": [1], "h": [1]},
#         },
#     "2000_01_04_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 12,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "c": [1], "d": [1], "h": [1]},
#         },
#     "2000_01_05_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 20,  "ora": 0, "minuto": 0,
#             "basket": {"c": [1], "d": [1], "e": [1], "f": [1], "g": [1]},
#         },
#     "2000_01_06_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 25,  "ora": 0, "minuto": 0,
#             "basket": {"e": [1], "f": [1], "g": [1]},
#         },
#     "2000_01_07_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 27,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "c": [1], "g": [1], "h": [1]},
#         },
#     "2000_01_09_00_00": {
#             "anno": 2000, "mese_n": 1, "giorno_n": 29,  "ora": 0, "minuto": 0,
#             "basket": {"c": [1], "d": [1]},
#         },
#     "2000_01_10_00_00": {
#             "anno": 2000, "mese_n": 2, "giorno_n": 3,  "ora": 0, "minuto": 0,
#             "basket": {"c": [1], "d": [1], "e": [1], "f": [1], "r": [1]},
#         },
#     "2000_01_11_00_00": {
#             "anno": 2000, "mese_n": 2, "giorno_n": 5,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "e": [1], "f": [1], "h": [1]},
#         },
#     "2000_01_12_00_00": {
#             "anno": 2000, "mese_n": 2, "giorno_n": 9,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "c": [1], "d": [1], "e": [1], "f": [1], "g": [1], "h": [1]},
#         },
#     "2000_01_14_00_00": {
#             "anno": 2000, "mese_n": 2, "giorno_n": 14,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "b": [1], "g": [1], "h": [1], "r": [1]},
#         },
#     "2000_01_15_00_00": {
#             "anno": 2000, "mese_n": 2, "giorno_n": 19,  "ora": 0, "minuto": 0,
#             "basket": {"a": [1], "c": [1], "d": [1]},
#         },
#     }
# }
#
# customers_data = {0: customer_data}
#
# customers_train_set, customers_test_set = split_train_test(customers_data,
#                                                                split_mode='loo',
#                                                                min_number_of_basket=10,
#                                                                min_basket_size=1,
#                                                                max_basket_size=float('inf'),
#                                                                min_item_occurrences=2,
#                                                                item2category=None)
#
# customers_data, new2old, old2new = remap_items_with_data(customers_train_set)
#
# for basket_id in sorted(customers_data[0]['data']):
#     print basket_id, customers_data[0]['data'][basket_id]['basket']
#
# print '\nnew2old', new2old
#
# tree = TARSTree(customers_data[0], root_value=None, root_count=None, root_timeseries=None)
#
# freq_rec_patterns = tree.mine_patterns(max_rec_dept=1)
#
# print len(freq_rec_patterns)





# customer_id = 16387 #16386
#
# print datetime.datetime.now(), 'build tree start'
# tree = TARSTree(customers_data[customer_id], root_value=None, root_count=None, root_timeseries=None)
# print datetime.datetime.now(), 'build tree end'
#
# print 'len(tree.item_support)', len(tree.item_support)
#
# print datetime.datetime.now(), 'mine patterns start'
# freq_rec_patterns = tree.mine_patterns(max_rec_dept=1)
# print datetime.datetime.now(), 'mine pattern end'
#
# print 'len(freq_rec_patterns)', len(freq_rec_patterns)

# print datetime.datetime.now(), 'mine rp_intervals_support end'
# rp_intervals_support = calculate_intervals_support(freq_rec_patterns, tree)
# print datetime.datetime.now(), 'mine rp_intervals_support end'
#
# next_baskets = customers_test_set[customer_id]['data']
# for next_basket_id in next_baskets:
#     day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')
#
#     rp_purchases, rp_day_of_last_purchase = calcualte_active_rp(
#         customer_id, customers_data, rp_intervals_support, day_of_next_purchase)
#
# print 'len(rp_purchases)', len(rp_purchases)
#
# print rp_purchases.keys()[:10]

# path = '/Users/riccardo/Documents/PhD/NextBasket/Dataset/'
#
# customers_data = read_data(path + 'dataset100.json')
# item2category = get_item2category(path + 'coop_categories_map.csv', category_index['categoria'])
#
# customers_train_set, customers_test_set = split_train_test(customers_data,
#                                                                split_mode='loo',
#                                                                min_number_of_basket=10,
#                                                                min_basket_size=1,
#                                                                max_basket_size=float('inf'),
#                                                                min_item_occurrences=2,
#                                                                item2category=item2category)
#
# customers_data, new2old, old2new = remap_items_with_data(customers_train_set)
#
# print 'test 10 utenti'
# performances_sup = list()
# performances_new = list()
#
# run_time = list()
# print datetime.datetime.now(), 'start at'
# for i, customer_id in enumerate(customers_data):
#
#     print datetime.datetime.now(), customer_id
#
#     # if i % 10 == 0:
#     #     print datetime.datetime.now(), 100.0 * i / len(customers_data)
#
#
#     tree = TARSTree(customers_data[customer_id], root_value=None, root_count=None, root_timeseries=None)
#     freq_rec_patterns = tree.mine_patterns(max_rec_dept=0)
#     rp_intervals_support = calculate_intervals_support(freq_rec_patterns, tree)
#
#     next_baskets = customers_test_set[customer_id]['data']
#     for next_basket_id in next_baskets:
#         day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')
#
#         start_time = datetime.datetime.now()
#         # rp_purchases, rp_day_of_last_purchase = calcualte_active_rp(
#         #     customers_data[customer_id]['data'], rp_intervals_support, day_of_next_purchase)
#         #
#         # freq_rec_patterns = tree.mine_patterns(max_rec_dept=1, patterns_subset=rp_purchases, nbr_patterns=None,
#         #                                        get_items_in_order_of_occurrences=False)
#         # rp_intervals_support = calculate_intervals_support(freq_rec_patterns, tree)
#
#         customer_data = customers_data[customer_id]['data']
#
#         rp_purchases, rp_day_of_last_purchase = calcualte_active_rp(
#             customer_data, rp_intervals_support, day_of_next_purchase)
#         # print len(rp_purchases)
#
#         item_score = calcualte_item_score(tree, rp_purchases, rp_intervals_support)
#         end_time = datetime.datetime.now()
#         run_time.append((end_time - start_time).total_seconds())
#
#         # if len(item_score) == 0:
#         #     print 'a'
#         #     continue
#
#         pred_len = 5
#         # pred_basket_sup = sorted(tree.single_item_support, key=tree.single_item_support.get,
#         #                          reverse=True)[:pred_len]
#         pred_basket_new = sorted(item_score, key=item_score.get, reverse=True)[:pred_len]
#
#         # pred_basket_sup = set([new2old[item] for item in pred_basket_sup])
#         pred_basket_new = set([new2old[item] for item in pred_basket_new])
#
#         next_basket = next_baskets[next_basket_id]['basket']
#         next_basket = set(next_basket.keys())
#
#         # evaluation = evaluate_prediction(next_basket, pred_basket_sup)
#         # performances_sup.append(evaluation['f1_score'])
#
#         evaluation = evaluate_prediction(next_basket, pred_basket_new)
#         performances_new.append(evaluation['f1_score'])
#
#     if i == 2:
#         break
#
# print datetime.datetime.now(), 'end at'
# print 'new', np.mean(performances_new)

# print 'sup, new', np.mean(performances_sup), np.mean(performances_new)

# print ''
#
# for k, v in calculate_aggregate(run_time).iteritems():
#     print k, v


def main():
    print 'TARS Test'

    pred_length = 5
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
        # item2category = None
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

    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print datetime.datetime.now(), 'Customers for test', len(customers_train_set), \
        '%.2f%%' % (100.0*len(customers_train_set)/len(customers_data))

    print datetime.datetime.now(), 'Create and build models'
    customers_recsys = dict()
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set.keys():
        print datetime.datetime.now(), customer_id
        if customer_id == 16483:
            continue
        customer_train_set = customers_train_set[customer_id]
        tars_tree = TARSTree(customer_train_set, root_value=None, root_count=None, root_timeseries=None)
        tars = tars_tree.mine_patterns(max_rec_dept=0, patterns_subset=None, nbr_patterns=None,
                                       get_items_in_order_of_occurrences=True)
        rs_intervals_support = calculate_intervals_support(tars, tars_tree)
        customers_recsys[customer_id] = (tars_tree, tars, rs_intervals_support)
        # break

    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Models built in', end_time - start_time

    print datetime.datetime.now(), 'Perform predictions'
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        print datetime.datetime.now(), customer_id
        if customer_id not in customers_recsys:
            continue

        tars_tree, tars, rs_intervals_support = customers_recsys[customer_id]
        customer_data = customers_train_set[customer_id]['data']

        next_baskets = customers_test_set[customer_id]['data']

        for next_basket_id in next_baskets:
            day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')

            rs_purchases, rs_day_of_last_purchase = calcualte_active_rp(customer_data, rs_intervals_support,
                                                                        day_of_next_purchase)

            tars = tars_tree.mine_patterns(max_rec_dept=1, patterns_subset=rs_purchases, nbr_patterns=None,
                                           get_items_in_order_of_occurrences=False)
            rs_intervals_support = calculate_intervals_support(tars, tars_tree)

            rs_purchases, rs_day_of_last_purchase = calcualte_active_rp(customer_data, rs_intervals_support,
                                                                        day_of_next_purchase)

            item_score = calcualte_item_score(tars_tree, rs_purchases, rs_intervals_support)

            pred_basket = sorted(item_score, key=item_score.get, reverse=True)[:pred_length]
            pred_basket = set([new2old[item] for item in pred_basket])
            # print pred_basket

            next_basket = next_baskets[next_basket_id]['basket']
            next_basket = set(next_basket.keys())
            # print next_basket

            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)

    end_time = datetime.datetime.now()
    print datetime.datetime.now(), 'Prediction performed in', end_time - start_time

    f1_values = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print datetime.datetime.now(), 'TARS', 'avg', stats['avg']
    print datetime.datetime.now(), 'TARS', 'iqm', stats['iqm']


if __name__ == "__main__":
    main()
