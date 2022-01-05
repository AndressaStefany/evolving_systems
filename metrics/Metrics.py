import numpy as np
from sklearn.metrics import cluster, precision_score, recall_score, f1_score, silhouette_score, adjusted_rand_score


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro')


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro')


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def silhouette(X, y_pred):
    # For when the labels are not known and the clusters are well defined
    return silhouette_score(X, y_pred, metric='euclidean')


def adjusted_rand(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)
