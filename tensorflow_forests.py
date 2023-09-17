import tensorflow as tf
import tensorflow_decision_forests as tfdf
from data import distance, train_valid_split
from math import exp

train, valid, y_valid = train_valid_split()
train_ds_pd = train.select_dtypes(include=['float64', 'int64'])
valid_ds_pd = valid.select_dtypes(include=['float64', 'int64'])

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

label = 'SalePriceLog'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, task=tfdf.keras.Task.REGRESSION)

# RandomForestModel/ GradientBoostedTreesModel/ CartModel
# model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
model.compile(metrics=["mse"])

model.fit(x=train_ds)
print(model.summary())
inspector = model.make_inspector()
print(inspector.evaluation())
evaluation = model.evaluate(x=valid_ds, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)
print(inspector.variable_importances()["NUM_AS_ROOT"])

preds = model.predict(valid_ds)
valid_ds_pd['PredictedPriceLog'] = preds.squeeze()
valid_ds_pd['PredictedPrice'] = valid_ds_pd['PredictedPriceLog'].apply(exp)
valid_ds_pd['SalePriceLog'] = y_valid
d: float = distance(valid_ds_pd)
print('distance: %.4f' % d)
