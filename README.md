# Purpose
Playground for the Real Estate Kaggle competition: 
    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# Launch
```
source ./venv/bin/activate
```

# Results
* All Houses with a price 20,000: Distance = 37.042
* All Houses with a price = mean: Distance = 7.1158
* All Houses with a price = SqFtMean x area: Distance = 8.6200
* TensorFlow: RandomForestModel: Distance = 2.2830
* TensorFlow: GradientBoostedTreesModel: Distance = 1.6900
* TensorFlow: CartModel: Distance = 3.4257
* TensorFlow: GradientBoostedTreesModel: Distance = Nope
* Multi Linear Regression: Distance: Distance: 11.0667
* Multi Linear Regression Lasso: Distance: Distance: 11.0681
* Multi Linear Regression Ridge: Distance: Distance: 11.0682
* Multi Linear Regression RidgeCV: Distance: Distance: 11.0686
* Multi Linear Regression SGDRegressor: Distance: Distance: 229.5353
* ElasticNet: 11.0439
* ElasticNetCV: 3.94
* Lars: 49.0597
* LarsCV: 11.9613
* OrthogonalMatchingPursuit: 3.8870
* ARDRegression: 11.0675
* BayesianRidge: 11.0627
* HuberRegressor: 12.3708
* QuantileRegressor: 3.5069
* RANSACRegressor: 11.1617
* TheilSenRegressor: 92.1564
* PoissonRegressor: 7.1158

# TODO:
* Remove outlyers?
* Enrich the data: 
- Dates and time to last transactions
- Localisation should be a string
- Add Price per sqft to the inputs 
* Commit to GitHub
* Regression
* Neuronal Network: https://realpython.com/python-ai-neural-network/ 
* FaceBook/ Meta API
* tensorflow/keras

TensorFlow: INVALID_ARGUMENT: 
The Distributed Gradient Boosted Tree learner does not support training from in-memory datasets 
(i.e. VerticalDataset in Yggdrasil Decision Forests, (non distributed) Dataset in TensorFlow Decision Forests). 
If your dataset is small, use the (non distributed) Gradient Boosted Tree learner 
(i.e. GRADIENT_BOOSTED_TREES with Yggdrasil Decision Forests, GradientBoostedTreesModel in TensorFlow Decision Forests). 
If your dataset is large, provide the dataset as a path (Yggdrasil Decision Forests) 
or use a TF Distribution Strategy (TensorFlow Decision Forests). [Op:SimpleMLCheckStatus] name: 

