import pandas as pd

train = pd.read_parquet('./data/train_sort.parquet')
sessionId = train['sessionId']
dups = train[sessionId.duplicated(keep=False)]
dups = dups.sort_values(by=['sessionId', 'visitStartTime'])
prev_ = dups[::2]
next_ = dups[1::2]

prev_ifbuy = (prev_['totals_transactionRevenue'] > 0).values
next_ifbuy = (next_['totals_transactionRevenue'] > 0).values
ifbuy = prev_ifbuy + next_ifbuy
prev_nobuy = prev_[~ifbuy]
next_nobuy = next_[~ifbuy]

train.drop(next_nobuy.index, inplace=True)
train.drop(prev_[ifbuy].index[~prev_ifbuy[ifbuy]], inplace=True)
train.drop(next_[ifbuy].index[~next_ifbuy[ifbuy]], inplace=True)

train.to_parquet('./data/train_sort.parquet')
