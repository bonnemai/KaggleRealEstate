from sklearn.neural_network import MLPRegressor
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
from math import exp
from data import get_train, get_valid, distance
field='SalePriceLog'
# X, y = make_regression(n_samples=200, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
train_df = get_train()
valid_df = get_valid()
train_df = train_df.select_dtypes(include=['float64', 'int64']).fillna(0)
valid_df = valid_df.select_dtypes(include=['float64', 'int64']).fillna(0)
y_train = train_df.pop(field)
regr = MLPRegressor(random_state=5, max_iter=500).fit(train_df, y_train)
valid_df.pop('SalePrice')
sale_price_log=valid_df.pop('SalePriceLog')
prediction = regr.predict(valid_df)
print('score: %.4f' % regr.score(valid_df, sale_price_log))
# print(prediction)
valid_df['SalePriceLog']=sale_price_log
valid_df['PredictedPriceLog'] = prediction
valid_df['PredictedPrice'] = valid_df['PredictedPriceLog'].apply(exp)
# valid_df[field]=sale_price

print('Distance: %.4f' % distance(valid_df))
