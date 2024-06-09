import pandas as pd
from utils.prepare_data import calc_feat_solo

def create_train_val_noagg(num: int):
    train = pd.read_parquet('./data/train_sort.parquet')
    test = pd.read_parquet('./data/test_sort.parquet')

    test_size = test.shape[0]
    split_date = train.iloc[-test_size]['date']
    val = train[train['date'] >= split_date]
    train = train[train['date'] < split_date]
    columns_to_drop = ['date', 'fullVisitorId', 'sessionId', 'visitId', 
                    'visitStartTime', 'geoNetwork_country', 'geoNetwork_region', 'geoNetwork_networkDomain', 
                    'geoNetwork_metro', 'geoNetwork_city', 'trafficSource_keyword', 
                    'trafficSource_referralPath', 'trafficSource_adwordsClickInfo.page',
                    'trafficSource_adwordsClickInfo.slot', 'trafficSource_adwordsClickInfo.gclId',
                    'trafficSource_adwordsClickInfo.adNetworkType', 'trafficSource_adContent', 'trafficSource_campaign']

    tr, vl = calc_feat_solo(train.copy(deep=True), val.copy(deep=True), columns_to_drop)
    tr.to_parquet(f'./data/train{num}.parquet')
    vl.to_parquet(f'./data/val{num}.parquet')
