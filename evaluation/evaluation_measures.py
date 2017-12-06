import numpy as np

def precision_score(real_basket, pred_basket):

    if len(pred_basket) == 0:
        return 0.0

    precision = 1.0 * len(real_basket & pred_basket) / len(pred_basket)

    return precision


def recall_score(real_basket, pred_basket):

    if len(real_basket) == 0:
        return 0.0

    recall = 1.0 * len(real_basket & pred_basket) / len(real_basket)

    return recall


def fbeta_score(real_basket, pred_basket, beta=1.0):
    precision = precision_score(real_basket, pred_basket)
    recall = recall_score(real_basket, pred_basket)

    if precision == 0 and recall == 0:
        return 0.0

    f_beta = 1.0 * (1 + beta) * precision * recall / ((beta**2 * precision) + recall)
    return f_beta


def f1_score(real_basket, pred_basket):
    return fbeta_score(real_basket, pred_basket, beta=1.0)


def f05_score(real_basket, pred_basket):
    return fbeta_score(real_basket, pred_basket, beta=0.5)


def f2_score(real_basket, pred_basket):
    return fbeta_score(real_basket, pred_basket, beta=2.0)


def hit_score(real_basket, pred_basket):
    return 1.0 if len(real_basket & pred_basket) else 0.0


def dcg(pred_basket_sup, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(pred_basket_sup)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcg(pred_basket_sup, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    best_dcg = dcg(sorted(pred_basket_sup, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(pred_basket_sup, rank) / best_dcg


def evaluate_prediction(real_basket, pred_basket):
    evaluation = {
        'precision': precision_score(real_basket, pred_basket),
        'recall': recall_score(real_basket, pred_basket),
        'f1_score': f1_score(real_basket, pred_basket),
        'f05_score': f05_score(real_basket, pred_basket),
        'f2_score': f2_score(real_basket, pred_basket),
        'hit_score': hit_score(real_basket, pred_basket),
    }
    return evaluation



def main():
    print 'Test Evaluation Measures'

    real_basket = [1, 2, 3, 4, 5, 6, 7, 8]
    pred_basket = [1,2,3, 4]

    real_basket = set(real_basket)
    pred_basket = set(pred_basket)

    print 'precision', precision_score(real_basket, pred_basket)
    print 'recall', recall_score(real_basket, pred_basket)
    print 'f1_score', f1_score(real_basket, pred_basket)
    print 'f05_score', f05_score(real_basket, pred_basket)
    print 'f2_score', f2_score(real_basket, pred_basket)
    print 'hit_score', hit_score(real_basket, pred_basket)

    # f05 = 1 li ho presi tutti ma forse li ho suggeriti molti di piu
    # f2 = 1, quelli che ho suggerito sono corretti ma forse l'ho suggerito uno solo
    # f1 via di mezzo


if __name__ == "__main__":
    main()
