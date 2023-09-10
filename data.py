from pandas import read_csv, DataFrame
from math import log, sqrt
from datetime import date

use_log: bool = True


def get_data() -> DataFrame:
    df = read_csv('./data/train.csv')
    df['PricePerLotArea'] = df['SalePrice'] / df['LotArea']
    df['DateSold1'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 1), axis=1)
    df['DateSold15'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 15), axis=1)
    df['DateSold28'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 28), axis=1)
    max_date: date = date(2010, 7, 1)
    df['nbDays'] = df.apply(lambda x: (x['DateSold1'] - max_date).days, axis=1)
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


if __name__ == '__main__':
    # df=get_data()
    # print('Sample:', len(df), df.head())
    df = get_valid()
    df['PredictedPrice'] = 20000
    d = distance(df)
    print('Sample:', len(df), 'Distance:', d, df['DateSold1'].max(), df['DateSold15'].min())
# df=get_re::maining()
# print('Sample:', len(df), df.head())
