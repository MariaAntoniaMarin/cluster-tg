import numpy as np
import pandas as pd
import random
import math


def euclideanDistance(x, y, c):
    return round(math.sqrt(((x-c[0])**2) + ((y-c[1])**2)), 2)


class KMeans():

    @staticmethod
    def cluster(array):
        min_ = min(array)
        for i, j in zip(array, range(len(array))):
            if min_ == i:
                return j+1

    @staticmethod
    def clusterKMeans(clusters_number, data):
        centroid_flag = True
        counter = 0
        previous_centroids = Centroids._random(clusters_number, data)
        ecli_dist_df = pd.DataFrame()
        cluster_df = pd.DataFrame({'x': data[:, 0],
                                   'y': data[:, 1]})
        while centroid_flag or counter < 1000:
            for i in range(clusters_number):
                ecli_dist_df['d{}'.format(i+1)] = [euclideanDistance(cluster_df['x'][j],
                                                                     cluster_df['y'][j], previous_centroids[i]) for j in range(len(cluster_df))]
            cluster_df['cluster'] = [
                KMeans.cluster(np.array(ecli_dist_df.iloc[i, :])) for i in range(len(cluster_df))]
            centroids = Centroids.calculate(clusters_number, cluster_df)
            centroid_check = Centroids.check(previous_centroids, centroids)
            if (centroid_check == False):
                previous_centroids = centroids
            else:
                centroid_flag = False
            counter += 1
            
            return cluster_df, centroids


class Centroids():

    @staticmethod
    def calculate(clusters_number, data):
        return np.array([[np.mean(data.loc[data['cluster'] == i+1, ['x']]), np.mean(data.loc[data['cluster'] == i+1, ['y']])] for i in range(clusters_number)])

    @staticmethod
    def check(previous_centroids, new_centroids):
        check_list = list()
        for i, j in zip(previous_centroids, new_centroids):
            if i[0] == j[0] and i[1] == j[1]:
                check_list.append(True)
            else:
                check_list.append(False)
        return all(check_list)

    @staticmethod
    def _random(clusters_number, data):
        return np.array([random.choice(data) for cluster in range(clusters_number)])
