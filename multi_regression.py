
from data import get_valid, get_train, distance
from sklearn import linear_model

train_ds_pd = get_train()
valid_ds_pd = get_valid()
train_ds_pd = train_ds_pd.select_dtypes(include=['float64', 'int64']).fillna(0)
valid_ds_pd = valid_ds_pd.select_dtypes(include=['float64', 'int64']).fillna(0)
# //tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(get_train(), label="train")
#
y=train_ds_pd.pop('SalePrice')
# regr = linear_model.LinearRegression()
# regr = linear_model.Lasso(alpha=0.1)
# regr = linear_model.RidgeCV()
# regr = linear_model.SGDRegressor()
# regr = linear_model.ElasticNetCV()
# regr = linear_model.OrthogonalMatchingPursuit()
regr = linear_model.PoissonRegressor()




regr.fit(train_ds_pd, y)
y_valid=valid_ds_pd.pop('SalePrice')
valid_ds_pd['PredictedPrice'] = regr.predict(valid_ds_pd)
valid_ds_pd['SalePrice']=y_valid
print("Distance: %.4f" % distance(valid_ds_pd))
# print("TensorFlow Decision Forests v" + tfdf.__version__)
#
#
#
# label = 'SalePrice'
# train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
# valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
#
# # model = tfdf.keras.RandomForestModel()
# # model.fit(train_ds)
# # print(model.summary())
#
# # RandomForestModel
# # GradientBoostedTreesModel
# # CartModel
# # DistributedGradientBoostedTreesModel
# rf = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
# # rf = tfdf.keras.DistributedGradientBoostedTreesModel( task=tfdf.keras.Task.REGRESSION)
#
# rf.compile(metrics=["mse"])
# rf.fit(x=train_ds)
# inspector = rf.make_inspector()
# inspector.evaluation()
# evaluation = rf.evaluate(x=valid_ds, return_dict=True)
# for name, value in evaluation.items():
#     print(f"{name}: {value:.4f}")
#
# print(f"Available variable importances:")
# for importance in inspector.variable_importances().keys():
#     print("\t", importance)
#
# inspector.variable_importances()["NUM_AS_ROOT"]
#
# test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
#     valid_ds_pd,
#     task=tfdf.keras.Task.REGRESSION)
#
# preds = rf.predict(test_ds)
# print('squeeze', preds.squeeze())
# valid_ds_pd['PredictedPrice']=preds.squeeze()
# d:float=distance(valid_ds_pd)
# print('distance: %.4f'% d)
# # output = pd.DataFrame({'Id': valid_ds_pd[['id']],
# #                        'SalePrice': preds.squeeze()})

# output.head()
