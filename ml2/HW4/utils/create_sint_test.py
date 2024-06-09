import numpy as np
import pandas as pd

from prepare_data import calc_feat_solo, calculate_feat_agg

train = pd.read_parquet('./data/train_sort.parquet')
test = pd.read_parquet('./data/test_sort.parquet')

train_new, test_new = calc_feat_solo(train.copy(deep=True), test.copy(deep=True), [])

data_all = pd.concat([train_new, test_new], axis=0).reset_index(drop=True)
data_all['totals_transactionRevenue'].fillna(0, inplace=True)
target = data_all['totals_transactionRevenue'].values
data_all.drop(data_all.index[data_all['totals_bounces']], inplace=True)

tst = data_all[data_all['date'] >= test_new['date'][0]]

test_sint = calculate_feat_agg(tst).reset_index()
test_sint['totals_transactionRevenue'] = np.nan
test_sint['ret'] = np.nan

test_sint.to_parquet('./data/sint_test.parquet')
