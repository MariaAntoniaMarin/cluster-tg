import pandas as pd
import re

from routers import router
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LON = re.findall("^lon\w{0,6}", 'lon')
LAT = re.findall("^lat\w{0,4}", 'lat')
_STORAGE_PATH = './data'


@router.post('/cluster/data/preprocessing/{filename}')
def preprocessing_data(filename: str):
    data = pd.read_csv(f'./uploaded_files/{filename}', sep=',', delimiter=',')
    data.dropna(subset=['lat', 'lon'], how='all', axis=0, inplace=True)
    coordinates = data.loc[:, ['lat', 'lon']].values
    complete_coordinates = SimpleImputer(
        strategy='mean').fit_transform(coordinates)
    # normalized_coordinates = StandardScaler().fit_transform(complete_coordinates)
    data = pd.DataFrame(complete_coordinates, columns=['lat', 'lon'])
    data = data[data['lat'].between(-90, 90) & data['lon'].between(-180, 180)]
    data.to_csv(f'{_STORAGE_PATH}/{filename}', index=False)
    return {'succesful': 'Created'}
