from pandas import read_csv, DataFrame, isnull
from math import log, sqrt
from datetime import date
from typing import List

use_log: bool = True
sale_price_log: str = 'SalePriceLog'


def to_int_list(df, field: str, target_field: str = 'PricePerLotArea'):
    df1 = df[[field, target_field]].groupby(field).mean()
    df1 = df1.sort_values(target_field)
    return list(df1.index)


def enrich_with_int(df: DataFrame) -> DataFrame:
    dtypes = df.select_dtypes(include=['object'])
    for dtype in dtypes:
        if 'Date' not in dtype:
            int_list: List[str] = to_int_list(df, dtype)
            df[f'{dtype}_int'] = df[dtype].apply(lambda x: -1 if isnull(x) else int_list.index(x))
    return df


def get_data() -> DataFrame:
    df: DataFrame = read_csv('./data/train.csv')
    df['PricePerLotArea'] = df['SalePrice'] / df['LotArea']
    df['DateSold1'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 1), axis=1)
    df['DateSold15'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 15), axis=1)
    df['DateSold28'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 28), axis=1)
    max_date: date = date(2010, 7, 1)
    df['nbDays'] = df.apply(lambda x: (x['DateSold1'] - max_date).days, axis=1)
    df: DataFrame = enrich_with_int(df)
    if use_log:
        df['SalePriceLog'] = df['SalePrice'].apply(log)  # Be careful... remove if from some regressions.
    return df.drop('Id', axis=1)


def get_valid(random_state: int = 5) -> DataFrame:
    df: DataFrame = get_data()
    return df.sample(frac=.2, random_state=random_state)


def get_train() -> DataFrame:
    df: DataFrame = get_data()
    sample: DataFrame = get_valid()
    df1: DataFrame = df[~df.index.isin(sample.index)].copy()
    return df1.drop('SalePrice', axis=1) if use_log else df1


def distance(df: DataFrame) -> float:
    if 'PredictedPrice' in df:
        df['PredictedPriceLog'] = df['PredictedPrice'].apply(lambda x: log(x) if x > 0 else 0)
    if 'SalePriceLog' not in df:
        df['SalePriceLog'] = df['SalePrice'].apply(log)
    df['Diff'] = df['PredictedPriceLog'] - df['SalePriceLog']
    mean: float = df['Diff'].mean()
    df['DiffSq'] = df['Diff'].apply(lambda x: (x - mean) * (x - mean))
    return sqrt(df['DiffSq'].sum())


def train_test_split():
    train = get_train()
    valid = get_valid()
    y_train = train.pop(sale_price_log)
    y_test = valid.pop(sale_price_log)
    return train, valid, y_train, y_test  # = train_test_split(data['data'], data['target'], test_size=.2)


if __name__ == '__main__':
    df = get_data()
    enrich_with_int(df)
    # list = to_int_list(df, 'MSZoning')
    # print('List: %s' % list)
    # print('Sample:', len(df), df.head())
    # df = get_valid()
    # df['PredictedPrice'] = 20000
    # d = distance(df)
    # print('Sample:', len(df), 'Distance:', d, df['DateSold1'].max(), df['DateSold15'].min())
# df=get_re::maining()
# print('Sample:', len(df), df.head())
