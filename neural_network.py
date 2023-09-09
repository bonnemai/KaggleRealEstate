from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from data import get_train, get_valid

X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
train_df=get_train()
valid_df=get_valid()
y_train=train_df.pop('SalePrice')
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
valid_df.pop('SalePrice')
prediction=regr.predict(valid_df)
print(regr.score(X_test, y_test))
print(prediction)