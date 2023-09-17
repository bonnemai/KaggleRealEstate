
from data import get_valid, get_train, distance
from sklearn import linear_model, neighbors

train_ds_pd = get_train()
valid_ds_pd = get_valid()
train_ds_pd = train_ds_pd.select_dtypes(include=['float64', 'int64']).fillna(0)
valid_ds_pd = valid_ds_pd.select_dtypes(include=['float64', 'int64']).fillna(0)
# //tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(get_train(), label="train")
#
y=train_ds_pd.pop('SalePriceLog')
# regr = linear_model.LinearRegression()
# regr = linear_model.Lasso(alpha=0.1)
# regr = linear_model.RidgeCV()
# regr = linear_model.SGDRegressor()
# regr = linear_model.ElasticNetCV()
# regr = linear_model.OrthogonalMatchingPursuit()
# regr = linear_model.PoissonRegressor()
regr = neighbors.KNeighborsRegressor(n_neighbors=7)

regr.fit(train_ds_pd, y)
valid_ds_pd.pop('SalePriceLog')
y_valid=valid_ds_pd.pop('SalePrice')
valid_ds_pd['PredictedPriceLog'] = regr.predict(valid_ds_pd)
valid_ds_pd['SalePrice']=y_valid
print("Distance: %.4f" % distance(valid_ds_pd))
