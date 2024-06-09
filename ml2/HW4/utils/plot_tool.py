import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional
from plotly.graph_objects import Figure

def create_stats_fig(df: pd.DataFrame, column_agg: str, plot_func, 
               sort_by: Optional[str] = None, limit: Optional[int] = None) -> Figure:
    """create plot with statistics

    Args:
        df (pd.DataFrame): dataframe
        column_agg (str): column by which statistics calculated
        plot_func (_type_): type of plot: bar, line, etc.
        sort_by (Optional[str], optional): sort plot by. Defaults to None.
        limit (Optional[int], optional): plot xaxis limitation. Defaults to None.

    Returns:
        Figure: plots with stats
    """
    column_names = ['counts', 'totalRevenue', 'nonzero', 'meanRevenue', 'nonzero_ratio', 
                'stdRevenue', 'meanRevenue_nonzero', 'stdRevenue_nonzero']

    stats = df.groupby(by=[column_agg]).agg(counts = ('totals_transactionRevenue', lambda x: len(x)),
                                            nonzero = ('totals_transactionRevenue', lambda x: np.sum(x != 0)),
                                            totalRevenue = ('totals_transactionRevenue', lambda x: np.sum(x)),
                                            meanRevenue = ('totals_transactionRevenue', lambda x: np.mean(x)),
                                            stdRevenue = ('totals_transactionRevenue', lambda x: np.std(x)),
                                            meanRevenue_nonzero = ('totals_transactionRevenue', lambda x: np.mean(x[x != 0])),
                                            stdRevenue_nonzero = ('totals_transactionRevenue', lambda x: np.std(x[x != 0]))
                                            )

    stats['nonzero_ratio'] = stats['nonzero'] / stats['counts']
    if sort_by:
        stats.sort_values(by=sort_by, ascending=False, inplace=True)
    if limit:
        stats = stats.iloc[:limit]

    fig = make_subplots(rows=4, cols=2, shared_xaxes=True, subplot_titles=column_names, 
                    horizontal_spacing=0.05, vertical_spacing=0.05)

    for i, column in enumerate(column_names):
        plot = plot_func(stats, x=stats.index, y=column)
        fig.add_trace(plot['data'][0], row=i // 2 + 1, col=i % 2 + 1)

    fig.update_layout(height=800, width=1200, title_text=column_agg+' stats')

    return fig

def compare_train_test(train: pd.DataFrame, test: pd.DataFrame, feature_name: str) -> Figure:
    """create plot with feature percent statistics to compare train and test

    Args:
        train (pd.DataFrame): train df
        test (pd.DataFrame): test df

    Returns:
        Figure: plot with stacked bars descriptions
    """
    train_count = train.groupby(feature_name).agg(count = ('fullVisitorId', 'count')) / train.shape[0]
    test_count = test.groupby(feature_name).agg(count = ('fullVisitorId', 'count')) / test.shape[0]
    count = train_count.join(test_count, rsuffix='_test', how='outer')
    count.columns = ['train', 'test']
    fig = px.bar(count, y=['train', 'test'], barmode='overlay')
    return fig
