from pandas import DataFrame

from data import get_train, get_valid, distance

def get_sqft(train:DataFrame, test:DataFrame):
	priceSqFt:float=train['PricePerLotArea'].mean()
	test['PredictedPrice']=test['LotArea'] * priceSqFt
	d:float=distance(test)
	print('SqFt: Distance: %.4f'% d)

def get_mean(train:DataFrame, test:DataFrame):
	mean:float=train['SalePrice'].mean()
	test['PredictedPrice']=mean
	d:float=distance(test)
	print('Mean: Distance: %.4f'% d)


if __name__== '__main__':
	get_mean(get_train(), get_valid())
	get_sqft(get_train(), get_valid())
