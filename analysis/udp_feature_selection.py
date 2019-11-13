"""
analysis/udp_feature_selection.py
author: zachandfox
date created: 2019-11-12
"""

import analysis.feature_selection as fs
#from CONSTANTS import CURRENT_YEAR
from __tpsdata__ import project_directory
import os
import pandas as pd
#import numpy as np

###### Import Data ######
# Identify Base file CSV's #
df_names = ['base_cost_val_mean', 'base_cost_val_median', 'base_mkt_val_mean', 'base_mkt_val_median', 'base_salep_mean', 'base_salep_median']
dfs = {file: pd.read_csv(project_directory+'1_analytical_files/{}.csv'.format(file), index_col='tract') for file in df_names}

###### Prep data for analysis ######
# Specify year range (and target year) #
start_year = 2013
target_year = 2018

# Keep columns of interest 
inputs = ['tract_over_city_{}'.format(year) for year in range(start_year, target_year+1)]
target = 'count_{}'.format(target_year)
columns =  + [target]

###### Run Analysis ######
results = {}

#results['RandomLasso'] = getTopFeaturesRandomLasso(df,'target')
results['Lasso'] = fs.getTopFeaturesLasso(df, target)
results['Ridge'] = fs.getTopFeaturesRidge(df,target)
results['Linear'] = fs.getTopFeaturesLinear(df, target)
results['RandomForest'] = fs.getTopFeaturesRandomForest(df,target)
results['Correlation'] = fs.getTopFeaturesF(df, target)
results['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df, target)))

results_df = pd.DataFrame(results)
methods = list(results.keys())
r = {}