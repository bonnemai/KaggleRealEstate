from sklearn.neural_network import MLPRegressor
from data import distance, train_valid_split

field='SalePriceLog'
is_prod: bool = False
train, valid, y_valid = train_valid_split(is_prod)
y_train = train.pop(field)
train_df = train.select_dtypes(include=['float64', 'int64']).fillna(0)
valid_df = valid.select_dtypes(include=['float64', 'int64']).fillna(0)

regr = MLPRegressor(random_state=5, max_iter=1500).fit(train_df, y_train)
sale_price_log=y_valid
prediction = regr.predict(valid_df)
print('score: %.4f' % regr.score(valid_df, sale_price_log))
valid_df['SalePriceLog']=sale_price_log
valid_df['PredictedPriceLog'] = prediction

print('Distance: %.4f' % distance(valid_df))
