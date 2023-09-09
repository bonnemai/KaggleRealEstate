from pandas import read_csv, DataFrame
from math import log, sqrt
from datetime import date


def get_data() -> DataFrame:
    df = read_csv('./data/train.csv')
    df['PricePerLotArea'] = df['SalePrice'] / df['LotArea']
    df['DateSold1'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 1), axis=1)
    df['DateSold15'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 15), axis=1)
    df['DateSold28'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 28), axis=1)
    max_date = date(2010, 7, 1)
    df['nbDays'] = df.apply(lambda x: (x['DateSold1'] - max_date).days, axis=1)

    return df.drop('Id', axis=1)


def get_valid(random_state: int = 5) -> DataFrame:
    df: DataFrame = get_data()
    return df.sample(frac=.2, random_state=random_state)


def get_train() -> DataFrame:
    df = get_data()
    sample = get_valid()
    df1 = df[~df.index.isin(sample.index)]
    return df1


def distance(df: DataFrame) -> float:
    df['PredictedPriceLog'] = df['PredictedPrice'].apply(lambda x: log(x) if x>0 else 0)
    df['SalePriceLog'] = df['SalePrice'].apply(log)
    df['Diff'] = df['PredictedPriceLog'] - df['SalePriceLog']
    df['DiffSq'] = df['Diff'].apply(lambda x: x * x)
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
