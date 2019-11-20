"""
Author: zachandfox

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
#from sklearn.linear_model import (LinearRegression, Ridge, 
#                                  Lasso, RandomizedLasso)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)
from sklearn.feature_selection import RFE, f_regression

from sklearn.datasets import load_diabetes

#REPOS = os.environ['REPOS']
#
data = load_diabetes()
#df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.DataFrame(data = np.c_[data['data'], data['target']], 
        columns = data['feature_names'] + ['target'])

def constructTopFeatureColumns(features):
    topFeatures = []
    for feature in features:
        if feature[0] > .009:
            topFeatures.append(feature[1])
    return topFeatures

def getTopFeaturesRF(df,predictor):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    #df['Gain'] = df['open']-df['close']
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)
    columns = X.columns
    #X = scaler_x.fit_transform(X)
    #y = scaler_y.fit_transform(y)
    # feature extraction
    model = RandomForestRegressor(n_estimators=300)
    model.fit(X, y)
    result = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), columns), reverse=True)
    return constructTopFeatureColumns(result),result

def getTopFeaturesRandomForest(df,predictor):
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)._get_numeric_data()
    columns = X.columns
    rfe = RandomForestRegressor(n_estimators=300)
    rfe.fit(X,y)
    return rank_to_dict(rfe.feature_importances_, columns)

def getTopFeaturesRidge(df,predictor):
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)._get_numeric_data()
    columns = X.columns
    ridge = Ridge(alpha=7)
    ridge.fit(X, y)
    return rank_to_dict(np.abs(ridge.coef_), columns)

def getTopFeaturesLasso(df,predictor):
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)._get_numeric_data()
    columns = X.columns
    lasso = Lasso(alpha=.05)
    lasso.fit(X, y)
    return rank_to_dict(np.abs(lasso.coef_), columns)

#def getTopFeaturesRandomLasso(df,predictor):
#    y = df[predictor]
#    #y = fixTime(y,scaler_y)
#    X = df.drop([predictor],axis=1)._get_numeric_data()
#    columns = X.columns
#    rlasso = RandomizedLasso(alpha=0.04)
#    rlasso.fit(X, y)
#    return rank_to_dict(np.abs(rlasso.scores_), columns)

def getTopFeaturesLinear(df,predictor):
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)._get_numeric_data()
    columns = X.columns
    lr = LinearRegression(normalize=True)
    lr.fit(X, y)
    return rank_to_dict(np.abs(lr.coef_), columns)

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def getTopFeaturesF(df,predictor):
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)._get_numeric_data()
    columns = X.columns
    f, pval  = f_regression(X, y, center=True)
    f[np.isnan(f)] = 0
    return rank_to_dict(f, columns)

def getTopFeaturesRFE(df,predictor):
    y = df[predictor]
    #y = fixTime(y,scaler_y)
    X = df.drop([predictor],axis=1)._get_numeric_data()
    columns = X.columns
    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,y)
    return list(map(float, rfe.ranking_))
    #return map(float, rfe.ranking_)
    #return dict(map(float, rfe.ranking_))

## Grab list of column headers
names = list(df.drop(['target'], axis=1)._get_numeric_data().columns)

results = {}

#results['RandomLasso'] = getTopFeaturesRandomLasso(df,'target')
results['Lasso'] = getTopFeaturesLasso(df,'target')
results['Ridge'] = getTopFeaturesRidge(df,'target')
results['Linear'] = getTopFeaturesLinear(df,'target')
results['RandomForest'] = getTopFeaturesRandomForest(df,'target')
results['Correlation'] = getTopFeaturesF(df,'target')
results['RFE'] = dict(zip(names,getTopFeaturesRFE(df,'target')))

results_df = pd.DataFrame(results)
methods = list(results.keys())
r = {}
#for name in names:
#    r[name] = round(np.mean([results[method][name] for method in results.keys()]), 2)
#for name in names:
#    for method in methods:
#        r[name] = results[method][name]
#    r[name] = round(np.mean([results[method][name] for method in results.keys()]), 2)
r = {name: round(np.mean([results[method][name] for method in results.keys()]), 2) for name in names}
 
methods = sorted(results.keys())
results["Mean"] = r
#       
#Output = pd.DataFrame.from_dict(results).sort_values(['Mean'],ascending=[0])
#Output.to_csv(REPOS+'/UDP/output/Ranking.csv', sep=',')
