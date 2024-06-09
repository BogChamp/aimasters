import numpy as np
import pandas as pd
from tqdm import tqdm

from prepare_data import calc_feat_solo, create_sintetic_train

train = pd.read_parquet('./data/train_sort.parquet')
test = pd.read_parquet('./data/test_sort.parquet')

train_new, test_new = calc_feat_solo(train.copy(deep=True), test.copy(deep=True), [])

data_all = pd.concat([train_new, test_new], axis=0).reset_index(drop=True)
data_all['totals_transactionRevenue'].fillna(0, inplace=True)
target = data_all['totals_transactionRevenue'].values
data_all.drop(data_all.index[data_all['totals_bounces']], inplace=True)

train_frame_width = 76
test_frame_width = 19
shift = 1

for shift in tqdm(range(4)):
    res_train = create_sintetic_train(data_all, shift+1, train_frame_width, test_frame_width)
    res_train.to_parquet(f'./data/sint_train{shift}.parquet')
