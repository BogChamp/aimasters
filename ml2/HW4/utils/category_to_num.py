import pandas as pd

def get_embeds1(df: pd.DataFrame):
    embs = df.values
    for i, column in enumerate(df.columns):
        if df[column].dtype == 'category':
            embs[:, i] = pd.factorize(df[column])[0]

    embs = embs.astype(float)
    return embs

def get_embeds2(df: pd.DataFrame):
    embs = df.copy(deep=True)
    for i, column in enumerate(embs.columns):
        if embs[column].dtype == 'category':
            embs = embs.join(pd.get_dummies(embs[column]), rsuffix=column)
            embs.drop(column, axis=1, inplace=True)

    embs = embs.astype(float)
    embs.columns = embs.columns.astype(str)
    return embs
