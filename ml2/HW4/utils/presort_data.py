import pandas as pd

train = pd.read_parquet('./data/train.parquet')
test = pd.read_parquet('./data/test.parquet')

train['totals_transactionRevenue'].fillna(0, inplace=True)

train_new = train.sort_values(by=['date', 'visitStartTime'])
train_new.reset_index(inplace=True)
train_new.drop('index', axis=1, inplace=True)

test_new = test.sort_values(by=['date', 'visitStartTime'])
test_new.reset_index(inplace=True)
test_new.drop('index', axis=1, inplace=True)

train_new.to_parquet('./data/train_sort.parquet')
test_new.to_parquet('./data/test_sort.parquet')
