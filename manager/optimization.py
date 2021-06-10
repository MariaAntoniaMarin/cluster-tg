import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans as km

from manager.cluster import KMeans
from manager.dunn_index import dunn


def silhouette(data, range_n_clusters):
    data_list = []
    for n_clusters in range_n_clusters:
        cluster_dataset, cluster_centroids = KMeans.clusterKMeans(
            n_clusters, data)
        silhouette_avg = silhouette_score(data, cluster_dataset['cluster'])
        single_dict = {'k': n_clusters, 'score': silhouette_avg}
        data_list.append(single_dict)
    
    return {'silhouette': data_list}
    


def davies_bouldin(data, range_n_clusters):
    data_list = []
    for n_clusters in range_n_clusters:
        cluster_dataset, cluster_centroids = KMeans.clusterKMeans(
            n_clusters, data)
        davies_b = davies_bouldin_score(data, cluster_dataset['cluster'])
        single_dict = {'k': n_clusters, 'score': davies_b}
        data_list.append(single_dict)

    return {'davies_bouldin': data_list}


def dunn_(data, range_n_clusters):
    data_list = []
    for n_clusters in range_n_clusters:
        k_means = km(n_clusters=n_clusters).fit_predict(data)
        d = euclidean_distances(data)
        _dunn = dunn(k_means, d, 'mean_cluster', 'nearest')
        single_dict = {'k': n_clusters, 'score': _dunn}
        data_list.append(single_dict)

    return {'davies_bouldin': data_list}