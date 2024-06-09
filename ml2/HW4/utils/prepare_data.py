import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

ZERO_LABEL = 'ZERO' # to group zero revenue categories
SMALL_LABEL = 'SMALL'

def calc_feat_solo(train: pd.DataFrame, test: pd.DataFrame, columns_to_drop: list[str]):
    train['date'] = pd.to_datetime(train['date'], format='%Y%m%d')
    test['date'] = pd.to_datetime(test['date'], format='%Y%m%d')

    # channelGrouping features
    cats, uniqs = pd.factorize(train['channelGrouping'])
    train['channelGrouping'] = cats
    cats_map = dict(zip(uniqs, range(len(uniqs))))
    test['channelGrouping'] = test['channelGrouping'].map(cats_map)
    train['channelGrouping'] = train['channelGrouping'].astype('category')
    test['channelGrouping'] = test['channelGrouping'].astype('category')

    # date features
    train['hours'] = pd.to_datetime(train['visitStartTime'], unit='s').dt.hour
    train['hours'] -= 8
    train.loc[train['hours'] < 0, 'hours'] += 24
    train['weekday'] = train['date'].dt.weekday
    train['is_wknd'] = train['weekday'] >= 5
    train['day_quarter'] = (train['hours'] // 4).astype('category')
    train['hours'] = train['hours'].astype(int)
    train['weekday'] = train['weekday'].astype('category')
    date_distance_tr = (train['date'].values - train['date'].values[-1]) / 1e9
    scaler = MinMaxScaler((0, 20))
    train['days_till_end'] = scaler.fit_transform(date_distance_tr.reshape(-1, 1)).reshape(-1, )

    test['hours'] = pd.to_datetime(test['visitStartTime'], unit='s').dt.hour
    test['hours'] -= 8
    test.loc[test['hours'] < 0, 'hours'] += 24
    test['weekday'] = test['date'].dt.weekday
    test['is_wknd'] = test['weekday'] >= 5
    test['day_quarter'] = (test['hours'] // 4).astype('category')
    test['hours'] = test['hours'].astype(int)
    test['weekday'] = test['weekday'].astype('category')
    date_distance_tst = (test['date'].values - test['date'].values[-1]) / 1e9
    test['days_till_end'] = scaler.transform(date_distance_tst.reshape(-1, 1)).reshape(-1, )
    

    # 'visitNumber'
    train['invvisNumber'] = 1 / train['visitNumber']
    test['invvisNumber'] = 1 / test['visitNumber']

    # device_browser features
    nonzero_info = train.groupby('device_browser').agg(nonzero = ('totals_transactionRevenue', lambda x: np.sum(x != 0)))
    zero_revenue_cats = np.isin(train['device_browser'].values, nonzero_info.index[nonzero_info.values.reshape(-1, ) == 0])
    train.loc[zero_revenue_cats, 'device_browser'] = ZERO_LABEL
    test_obs = test['device_browser'].unique()
    train_obs = train['device_browser'].unique()
    zero_revenue_cats = np.isin(test['device_browser'].values, test_obs[~np.isin(test_obs, train_obs)])
    test.loc[zero_revenue_cats, 'device_browser'] = ZERO_LABEL

    train['device_browser'] = train['device_browser'].astype('category')
    test['device_browser'] = test['device_browser'].astype('category')

    # device_operatingSystem features
    nonzero_info = train.groupby('device_operatingSystem').agg(nonzero = ('totals_transactionRevenue', lambda x: np.sum(x != 0)))
    zero_revenue_cats = np.isin(train['device_operatingSystem'].values, nonzero_info.index[nonzero_info.values.reshape(-1, ) == 0])
    train.loc[zero_revenue_cats, 'device_operatingSystem'] = ZERO_LABEL
    test_obs = test['device_operatingSystem'].unique()
    train_obs = train['device_operatingSystem'].unique()
    zero_revenue_cats = np.isin(test['device_operatingSystem'].values, test_obs[~np.isin(test_obs, train_obs)])
    test.loc[zero_revenue_cats, 'device_operatingSystem'] = ZERO_LABEL

    train['device_operatingSystem'] = train['device_operatingSystem'].astype('category')
    test['device_operatingSystem'] = test['device_operatingSystem'].astype('category')

    # device_deviceCategory
    train['device_deviceCategory'] = train['device_deviceCategory'].astype('category')
    test['device_deviceCategory'] = test['device_deviceCategory'].astype('category')

    # geoNetwork_continent
    train['geoNetwork_continent'] = train['geoNetwork_continent'].astype('category')
    test['geoNetwork_continent'] = test['geoNetwork_continent'].astype('category')

    # geoNetwork_subContinent
    nonzero_info = train.groupby('geoNetwork_subContinent').agg(nonzero = ('totals_transactionRevenue', lambda x: np.sum(x != 0)))
    zero_revenue_cats = np.isin(train['geoNetwork_subContinent'].values, nonzero_info.index[nonzero_info.values.reshape(-1, ) == 0])
    train.loc[zero_revenue_cats, 'geoNetwork_subContinent'] = ZERO_LABEL
    test_obs = test['geoNetwork_subContinent'].unique()
    train_obs = train['geoNetwork_subContinent'].unique()
    zero_revenue_cats = np.isin(test['geoNetwork_subContinent'].values, test_obs[~np.isin(test_obs, train_obs)])
    test.loc[zero_revenue_cats, 'geoNetwork_subContinent'] = ZERO_LABEL

    train['geoNetwork_subContinent'] = train['geoNetwork_subContinent'].astype('category')
    test['geoNetwork_subContinent'] = test['geoNetwork_subContinent'].astype('category')

    train['isNorthAm'] = train['geoNetwork_subContinent'] == 'Northern America'
    test['isNorthAm'] = test['geoNetwork_subContinent'] == 'Northern America'
    
    # geoNetwork_country
    train['isUS'] = train['geoNetwork_country'] == 'United States'
    train['isCanada'] = train['geoNetwork_country'] == 'Canada'
    train['isVenez'] = train['geoNetwork_country'] == 'Venezuela'

    test['isUS'] = test['geoNetwork_country'] == 'United States'
    test['isCanada'] = test['geoNetwork_country'] == 'Canada'
    test['isVenez'] = test['geoNetwork_country'] == 'Venezuela'

    # totals_hits
    # totals_pageviews
    train['totals_hits'] = train['totals_hits'].astype(int)
    test['totals_hits'] = test['totals_hits'].astype(int)

    train['totals_pageviews'].fillna('0', inplace=True)
    train['totals_pageviews'] = train['totals_pageviews'].astype(int)

    test['totals_pageviews'].fillna('0', inplace=True)
    test['totals_pageviews'] = test['totals_pageviews'].astype(int)

    # totals_bounces & totals_newVisits
    train['totals_bounces'] = train['totals_bounces'].astype(bool)
    train['totals_newVisits'] = train['totals_newVisits'].astype(bool)

    test['totals_bounces'] = test['totals_bounces'].astype(bool)
    test['totals_newVisits'] = test['totals_newVisits'].astype(bool)

    # traffic features
    train['trafficSource_isTrueDirect'] = train['trafficSource_isTrueDirect'].astype(bool)
    train['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    train['trafficSource_adwordsClickInfo.isVideoAd'] = train['trafficSource_adwordsClickInfo.isVideoAd'].astype(bool)

    test['trafficSource_isTrueDirect'] = test['trafficSource_isTrueDirect'].astype(bool)
    test['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    test['trafficSource_adwordsClickInfo.isVideoAd'] = test['trafficSource_adwordsClickInfo.isVideoAd'].astype(bool)

    # trafficSource_medium
    train['trafficSource_medium'] = train['trafficSource_medium'].astype('category')
    test['trafficSource_medium'] = test['trafficSource_medium'].astype('category')

    # trafficSource_source
    nonzero_info = train.groupby('trafficSource_source').agg(nonzero = ('totals_transactionRevenue', lambda x: np.sum(x != 0)))
    zero_revenue_cats = np.isin(train['trafficSource_source'].values, nonzero_info.index[nonzero_info.values.reshape(-1, ) == 0])
    train.loc[zero_revenue_cats, 'trafficSource_source'] = ZERO_LABEL
    test_obs = test['trafficSource_source'].unique()
    train_obs = train['trafficSource_source'].unique()
    zero_revenue_cats = np.isin(test['trafficSource_source'].values, test_obs[~np.isin(test_obs, train_obs)])
    test.loc[zero_revenue_cats, 'trafficSource_source'] = ZERO_LABEL

    small_ind = nonzero_info.index[((nonzero_info.values > 0) * (nonzero_info.values < 1000)).reshape(-1,)]
    small_revenue_cats = np.isin(train['trafficSource_source'].values, small_ind)
    train.loc[small_revenue_cats, 'trafficSource_source'] = SMALL_LABEL
    small_revenue_cats = np.isin(test['trafficSource_source'].values, small_ind)
    test.loc[small_revenue_cats, 'trafficSource_source'] = SMALL_LABEL

    big_ind = nonzero_info.index[(nonzero_info.values >= 1000).reshape(-1,)]
    big_revenue_cats = np.isin(train['trafficSource_source'].values, big_ind)
    train.loc[big_revenue_cats, 'trafficSource_source'] = 'BIG'
    big_revenue_cats = np.isin(test['trafficSource_source'].values, big_ind)
    test.loc[big_revenue_cats, 'trafficSource_source'] = 'BIG'

    train['trafficSource_source'] = train['trafficSource_source'].astype('category')
    test['trafficSource_source'] = test['trafficSource_source'].astype('category')

    # drop columns

    train.drop(columns_to_drop, axis=1, inplace=True)
    test.drop(columns_to_drop, axis=1, inplace=True)

    return train, test

def prepare_data_solo(tr: pd.DataFrame, vl: pd.DataFrame):
    tr.drop(tr.index[tr['totals_bounces'].values], inplace=True)
    vl.drop(vl.index[vl['totals_bounces'].values], inplace=True)
    tr.drop(['totals_bounces'], axis=1)
    vl.drop(['totals_bounces'], axis=1)
    tr.drop(tr.index[tr['channelGrouping'] == '(Other)'], inplace=True)
    vl.drop(vl.index[vl['channelGrouping'] == '(Other)'], inplace=True)
    tr.drop(tr.index[tr['device_browser'] == 'ZERO'], inplace=True)
    vl.drop(vl.index[vl['device_browser'] == 'ZERO'], inplace=True)
    tr.drop(tr.index[tr['device_operatingSystem'] == 'ZERO'], inplace=True)
    vl.drop(vl.index[vl['device_operatingSystem'] == 'ZERO'], inplace=True)
    tr.drop(tr.index[(tr['visitNumber'] >= 110)], inplace=True) # 30
    vl.drop(vl.index[(vl['visitNumber'] >= 110)], inplace=True)
    tr.drop(tr.index[(tr['totals_hits'] >= 200)], inplace=True) # 100
    vl.drop(vl.index[(vl['totals_hits'] >= 200)], inplace=True)
    tr.drop(tr.index[(tr['totals_pageviews'] >= 150)], inplace=True) # 100
    vl.drop(vl.index[(vl['totals_pageviews'] >= 150)], inplace=True)

    return tr, vl

def get_category_by_freq(x, top:int = 1):
    uniqs, freqs = np.unique(x, return_counts=True)
    inds = np.argsort(freqs)
    if len(uniqs) >= top:
        return uniqs[inds][-top]
    else:
        return np.nan

def calculate_feat_agg(data_df: pd.DataFrame):
    return data_df.groupby('fullVisitorId').agg(
        entries = ('channelGrouping', lambda x: len(x)),
        channels_num = ('channelGrouping', lambda x: len(set(x))),
        first_channel = ('channelGrouping', lambda x: x.iloc[0]), 
        channel_max_freq = ('channelGrouping', lambda x: get_category_by_freq(x, 1)),
        channel_max2_freq = ('channelGrouping', lambda x: get_category_by_freq(x, 2)),
        visit_number = ('visitNumber', lambda x: np.nanmax(x)),
        browser_max_freq = ('device_browser', lambda x: get_category_by_freq(x, 1)),
        browser_max2_freq = ('device_browser', lambda x: get_category_by_freq(x, 2)),
        browser_num = ('device_browser', lambda x: len(set(x))),
        OS_max_freq = ('device_operatingSystem', lambda x: get_category_by_freq(x, 1)),
        OS_max2_freq = ('device_operatingSystem', lambda x: get_category_by_freq(x, 2)),
        OS_num = ('device_operatingSystem', lambda x: len(set(x))),
        mobile_max_freq = ('device_isMobile', lambda x: get_category_by_freq(x, 1)),
        mobile_mean = ('device_isMobile', lambda x: np.nanmean(x)),
        device_max_freq = ('device_deviceCategory', lambda x: get_category_by_freq(x, 1)),
        device_num = ('device_deviceCategory', lambda x: len(set(x))),
        continent_max_freq = ('geoNetwork_continent', lambda x: get_category_by_freq(x, 1)),
        continent_max2_freq = ('geoNetwork_continent', lambda x: get_category_by_freq(x, 2)),
        continent_num = ('geoNetwork_continent', lambda x: len(set(x))),
        subcontinent_max_freq = ('geoNetwork_subContinent', lambda x: get_category_by_freq(x, 1)),
        subcontinent_max2_freq = ('geoNetwork_subContinent', lambda x: get_category_by_freq(x, 2)),
        subcontinent_num = ('geoNetwork_subContinent', lambda x: len(set(x))),
        country_num = ('geoNetwork_country', lambda x: len(set(x))),
        network_num = ('geoNetwork_networkDomain', lambda x: len(set(x))),
        network_hidden = ('geoNetwork_networkDomain', lambda x: np.nanmean(x == 'unknown.unknown')),
        network_not_set = ('geoNetwork_networkDomain', lambda x: np.nanmean(x == '(not set)')),
        hits_sum = ('totals_hits', lambda x: np.nansum(x)),
        hits_mean = ('totals_hits', lambda x: np.nanmean(x)),
        hits_std = ('totals_hits', lambda x: np.nanstd(x)),
        views_sum = ('totals_hits', lambda x: np.nansum(x)),
        views_mean = ('totals_pageviews', lambda x: np.nanmean(x)),
        views_std = ('totals_pageviews', lambda x: np.nanstd(x)),
        views_max = ('totals_pageviews', lambda x: np.nanmax(x)),
        views_min = ('totals_pageviews', lambda x: np.nanmin(x)),
        new_visit_max_freq = ('totals_newVisits', lambda x: get_category_by_freq(x, 1)),
        new_visit_mean = ('totals_newVisits', lambda x: np.nanmean(x)),
        first_new_visit = ('totals_newVisits', lambda x: x.iloc[0]), 
        source_max_freq = ('trafficSource_source', lambda x: get_category_by_freq(x, 1)),
        source_max2_freq = ('trafficSource_source', lambda x: get_category_by_freq(x, 2)),
        source_num = ('trafficSource_source', lambda x: len(set(x))),
        medium_max_freq = ('trafficSource_medium', lambda x: get_category_by_freq(x, 1)),
        medium_max2_freq = ('trafficSource_medium', lambda x: get_category_by_freq(x, 2)),
        medium_num = ('trafficSource_medium', lambda x: len(set(x))),
        has_keyword = ('trafficSource_keyword', lambda x: len(set(x))), 
        direct_mean = ('trafficSource_isTrueDirect', lambda x: np.nanmean(x)),
        direct_first = ('trafficSource_isTrueDirect', lambda x: x.iloc[0]),
        direct_max_freq = ('trafficSource_isTrueDirect', lambda x: get_category_by_freq(x, 1)),
        video_mean = ('trafficSource_adwordsClickInfo.isVideoAd', lambda x: np.nanmean(x)),
        video_first = ('trafficSource_adwordsClickInfo.isVideoAd', lambda x: x.iloc[0]),
        video_max_freq = ('trafficSource_adwordsClickInfo.isVideoAd', lambda x: get_category_by_freq(x, 1)),
        hours_max_freq = ('hours', lambda x: get_category_by_freq(x, 1)),
        weekday_max_freq = ('weekday', lambda x: get_category_by_freq(x, 1)),
        weekday_max2_freq = ('weekday', lambda x: get_category_by_freq(x, 2)),
        wknd_mean = ('is_wknd', lambda x: np.nanmean(x)),
        daytime_max_freq = ('day_quarter', lambda x: get_category_by_freq(x, 1)),
        daytime_max2_freq = ('day_quarter', lambda x: get_category_by_freq(x, 2)),
        daytime_num = ('day_quarter', lambda x: len(set(x))),
        invvis_mean = ('invvisNumber', lambda x: np.nanmean(x)),
        invvis_std = ('invvisNumber', lambda x: np.nanstd(x)),
        isNAM = ('isNorthAm', lambda x: np.nanmean(x)),
        isUS = ('isUS', lambda x: np.nanmean(x)),
        isCanada = ('isCanada', lambda x: np.nanmean(x)),
        isVenez = ('isVenez', lambda x: np.nanmean(x))
    )

def create_sintetic_train(dataset: pd.DataFrame, shift: int, 
                          train_frame_width: int, test_frame_width: int):
    min_date = min(dataset['date'])
    min_date_train = min_date + (shift-1)*timedelta(days=train_frame_width)
    time_frame_train = dataset[(dataset['date'] >= min_date_train) & \
                            (dataset['date'] < (min_date + shift*timedelta(days=train_frame_width)))]
    max_date_train = time_frame_train['date'].values[-1]

    min_date_test = min_date + shift*timedelta(days=train_frame_width)
    time_frame_test = dataset[(dataset['date'] >= min_date_test) & \
                            (dataset['date'] < (min_date + shift*timedelta(days=train_frame_width+test_frame_width)))]
    max_date_test = time_frame_test['date'].values[-1]

    future_users = set(time_frame_test['fullVisitorId'].values)
    returned_users = time_frame_train[time_frame_train['fullVisitorId'].isin(future_users)]
    time_frame_test = time_frame_test[time_frame_test['fullVisitorId'].isin(set(returned_users['fullVisitorId']))]

    target_frame = time_frame_test.groupby('fullVisitorId')[['totals_transactionRevenue']].sum().reset_index()
    target_frame['ret'] = 1

    not_returned_users = time_frame_train[~time_frame_train['fullVisitorId'].isin(future_users)]
    not_returned_users['totals_transactionRevenue'] = 0
    not_returned_users['ret'] = 0

    target_frame = pd.concat([target_frame, not_returned_users[['fullVisitorId', 'totals_transactionRevenue', 'ret']]], 
                            axis=0).reset_index(drop=True)
    
    feats = calculate_feat_agg(time_frame_train).reset_index()

    res_train = pd.merge(feats, target_frame, left_on='fullVisitorId', right_on='fullVisitorId')

    return res_train
