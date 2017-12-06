import numpy as np
import scipy.stats as stats


__author__ = 'Riccardo Guidotti'


def interquartile_range_mean(values):
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)

    iqr_values = list()
    for v in values:
        if p25 <= v <= p75:
            iqr_values.append(v)

    return np.mean(iqr_values)


def mean_absolute_deviation(values):
    median = np.mean(values)
    dev_values = [abs(v - median) for v in values]

    return np.mean(dev_values)


def gini_coefficient(values):
    sort_values = sorted(values)
    cum_values = np.cumsum(sort_values)
    return 1.0 + 1.0/len(values) - 2 * (sum(cum_values) / (cum_values[-1] * len(values)))


def normalize(x):
    v_min = min(x)
    v_max = max(x)
    u = []
    for e in x:
        if v_max - v_min != 0:
            u.append((e-v_min) / (v_max - v_min))
        else:
            u.append(1.0/len(x))
    return u


def normalize_dic(x):
    v_min = min(x.values())
    v_max = max(x.values())
    n = dict()
    for e in x:
        if v_max - v_min != 0:
            n[e] = (1.0*(x[e]-v_min) / (v_max - v_min))
        else:
            n[e] = (1.0/len(x))
    return n


def calculate_aggregate(values):
    agg_measures = {
        'avg': np.mean(values),
        'std': np.std(values),
        'var': np.var(values),
        'med': np.median(values),
        '10p': np.percentile(values, 10),
        '25p': np.percentile(values, 25),
        '50p': np.percentile(values, 50),
        '75p': np.percentile(values, 75),
        '90p': np.percentile(values, 90),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'iqm': interquartile_range_mean(values),
        'mad': mean_absolute_deviation(values),
        'cov': 1.0 * np.mean(values) / np.std(values),
        'gin': gini_coefficient(values),
        'skw': stats.skew(values),
        'kur': stats.kurtosis(values),
        'sum': np.sum(values)
    }

    return agg_measures
