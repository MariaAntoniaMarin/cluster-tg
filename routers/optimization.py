import pandas as pd
import json
import re

from routers import router

from manager.optimization import silhouette, davies_bouldin, dunn_
from manager.schemas.cluster import RangeClusterNumbers


_STORAGE_PATH_DICTIONARIES_OPTIMIZATION = './dictionaries/optimization'


@router.post('/cluster/optimization/silhouette/{filename}')
def silhouette_score(filename: str, rk: RangeClusterNumbers):
    data = pd.read_csv(f'./data/{filename}')
    range_n_clusters = range(rk.start if rk.start and rk.start > 1 else 2,
                             rk.stop if rk.stop and rk.stop > rk.start else rk.start+3,
                             rk.step if rk.step else 1)
    dictionary = silhouette(data.values, range_n_clusters)
    out_file = open(f'{_STORAGE_PATH_DICTIONARIES_OPTIMIZATION}/{filename}silhouette.json','w+')
    json.dump(dictionary, out_file, indent=4)
    return {'succesful': 'Created'}


@router.post('/cluster/optimization/davies-bouldin/{filename}')
def davies_bouldin_score(filename: str, rk: RangeClusterNumbers):
    data = pd.read_csv(f'./data/{filename}')
    range_n_clusters = range(rk.start if rk.start and rk.start > 1 else 2,
                             rk.stop if rk.stop and rk.stop > rk.start else rk.start+3,
                             rk.step if rk.step else 1)
    dictionary = davies_bouldin(data.values, range_n_clusters)
    out_file = open(f'{_STORAGE_PATH_DICTIONARIES_OPTIMIZATION}/{filename}davies.json','w+')
    json.dump(dictionary, out_file, indent=4)
    return {'succesful': 'Created'}


@router.post('/cluster/optimization/dunn/{filename}')
def dunn_score(filename: str, rk: RangeClusterNumbers):
    data = pd.read_csv(f'./data/{filename}')
    range_n_clusters = range(rk.start if rk.start and rk.start > 1 else 2,
                             rk.stop if rk.stop and rk.stop > rk.start else rk.start+3,
                             rk.step if rk.step else 1)
    dictionary = dunn_(data.values, range_n_clusters)
    out_file = open(f'{_STORAGE_PATH_DICTIONARIES_OPTIMIZATION}/{filename}duun.json','w+')
    json.dump(dictionary, out_file, indent=4)
    return {'succesful': 'Created'}
