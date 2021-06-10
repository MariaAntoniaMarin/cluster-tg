import pandas as pd
import random
import json

from routers import router

from manager.cluster import KMeans
from manager.schemas.cluster import ClusterNumber

_STORAGE_PATH_DATA = './data'
_STORAGE_PATH_DICTIONARIES = './dictionaries'


@router.post('/cluster/data/kmeans/{filename}')
def apply_cluster(filename: str, n_clus:ClusterNumber):
    data = pd.read_csv(f'{_STORAGE_PATH_DATA}/{filename}')
    dictionary = dict()
    cluster_dataset, cluster_centroids = KMeans.clusterKMeans(n_clus.k, data.values)
    cluster_dataset.to_csv(f'{_STORAGE_PATH_DATA}/{filename}', index=False)
    for k in range(1,n_clus.k+1):
        data = cluster_dataset.loc[cluster_dataset.cluster==k, ['x','y']]
        _dict = [{'x':x, 'y':y} for x, y in zip(data['x'], data['y'])]
        dictionary[f'clus_{k}'] = _dict
    out_file = open(f'{_STORAGE_PATH_DICTIONARIES}/{filename}.json','w+')
    json.dump(dictionary, out_file, indent=4)
    return {'succesful': 'Created'}
