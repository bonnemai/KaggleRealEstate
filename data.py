from pandas import read_csv, DataFrame, isnull, Series
from math import log, sqrt
from datetime import date
from typing import List, Mapping, Optional, Tuple

use_log: bool = True
sale_price_log: str = 'SalePriceLog'


def to_int_list(df, field: str, target_field: str = 'PricePerLotArea'):
    df1 = df[[field, target_field]].groupby(field).mean()
    df1 = df1.sort_values(target_field)
    return list(df1.index)


def get_int_list_map(df: DataFrame) -> Mapping[str, List[str]]:
    dtypes: List[str] = df.select_dtypes(include=['object'])
    results: Mapping[str, List[str]] = {}
    for dtype in dtypes:
        if 'Date' not in dtype:
            int_list: List[str] = to_int_list(df, dtype)
            results[dtype] = int_list
            # df[f'{dtype}_int'] = df[dtype].apply(lambda x: -1 if isnull(x) else int_list.index(x))
    return results


def enrich_with_int(df: DataFrame, int_list_map: Mapping[str, List[str]]) -> DataFrame:
    dtypes = df.select_dtypes(include=['object'])
    for dtype in dtypes:
        if 'Date' not in dtype:
            int_list: List[str] = int_list_map[dtype]
            df[f'{dtype}_int'] = df[dtype].apply(lambda x: -1 if isnull(x) or x not in int_list else int_list.index(x))
    return df


def get_data(type: str = 'train') -> DataFrame:
    df: DataFrame = read_csv(f'./data/{type}.csv')
    df['DateSold1'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 1), axis=1)
    df['DateSold15'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 15), axis=1)
    df['DateSold28'] = df.apply(lambda x: date(x['YrSold'], x['MoSold'], 28), axis=1)
    max_date: date = date(2010, 7, 1)
    df['nbDays'] = df.apply(lambda x: (x['DateSold1'] - max_date).days, axis=1)
    return df.drop('Id', axis=1)

def get_id(type: str = 'train') -> Series:
    df: DataFrame = read_csv(f'./data/{type}.csv')
    return df.pop('Id')

def get_valid(random_state: int = 5) -> DataFrame:
    """
    Doesn't container SalePrice
    """
    df: DataFrame = get_data()
    int_list_map = get_int_list_map(get_train())
    df_sample = df.sample(frac=.2, random_state=random_state)
    df_sample = enrich_with_int(df_sample, int_list_map)
    return df_sample


def get_train() -> DataFrame:
    """
    Has Sale Price
    """
    df: DataFrame = get_data()
    sample: DataFrame = get_valid()
    df1: DataFrame = df[~df.index.isin(sample.index)].copy()
    df1['PricePerLotArea'] = df['SalePrice'] / df['LotArea']
    int_list_map = get_int_list_map(df1)
    df1 = enrich_with_int(df1, int_list_map)
    if use_log:
        df1['SalePriceLog'] = df1['SalePrice'].apply(log)
    return df1.drop(['SalePrice', 'PricePerLotArea'], axis=1) if use_log else df1


def distance(df: DataFrame) -> float:
    if 'PredictedPrice' in df:
        df['PredictedPriceLog'] = df['PredictedPrice'].apply(lambda x: log(x) if x > 0 else 0)
    if 'SalePriceLog' not in df:
        df['SalePriceLog'] = df['SalePrice'].apply(log)
    df['Diff'] = df['PredictedPriceLog'] - df['SalePriceLog']
    mean: float = df['Diff'].mean()
    df['DiffSq'] = df['Diff'].apply(lambda x: (x - mean) * (x - mean))
    return sqrt(df['DiffSq'].sum())


def train_valid_split(is_prod: bool = False) -> Tuple[DataFrame, DataFrame, Optional[Series]]:
    random_state = 5
    df: DataFrame = get_data()
    if use_log:
        df['SalePriceLog'] = df['SalePrice'].apply(log)
    if is_prod:
        valid: DataFrame = get_data('test')
        train: DataFrame = df
    else:
        valid: DataFrame = df.sample(frac=.2, random_state=random_state)
        train: DataFrame = df[~df.index.isin(valid.index)].copy()
    train['PricePerLotArea'] = df['SalePrice'] / df['LotArea']
    int_list_map: Mapping[str, List[str]] = get_int_list_map(train)
    train: DataFrame = enrich_with_int(train, int_list_map)
    valid: DataFrame = enrich_with_int(valid, int_list_map)
    train = train.drop(['SalePrice', 'PricePerLotArea'], axis=1)
    if 'SalePrice' in valid:
        valid = valid.drop('SalePrice', axis=1)
    y_valid = None if is_prod else valid.pop('SalePriceLog')
    return train, valid, y_valid  # = train_test_split(data['data'], data['target'], test_size=.2)


if __name__ == '__main__':
    train, valid, y_valid = train_valid_split(True)
    print('train: %.0f valid %.0f ' % (len(train), len(valid)))
    # enrich_with_int(df)
    # list = to_int_list(df, 'MSZoning')
    # print('List: %s' % list)
    # print('Sample:', len(df), df.head())
    # df = get_valid()
    # df['PredictedPrice'] = 20000
    # d = distance(df)
    # print('Sample:', len(df), 'Distance:', d, df['DateSold1'].max(), df['DateSold15'].min())
# df=get_re::maining()
# print('Sample:', len(df), df.head())
