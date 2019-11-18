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
#df_names = ['base_cost_val_mean', 'base_cost_val_median', 'base_mkt_val_mean', 'base_mkt_val_median', 'base_salep_mean', 'base_salep_median']
#dfs = {file: pd.read_csv(project_directory+'1_analytical_files/{}.csv'.format(file), index_col='tract') for file in df_names}
df = pd.read_csv(project_directory+'1_analytical_files/base_file.csv')
###### Prep data for analysis ######
# Specify year range (and target year) #
start_year = 2013
target_year = 2018

# Keep columns of interest df
#inputs = ['count_{}'.format(year) for year in range(start_year, target_year)] + ['salep_{}'.format(year) for year in range(start_year, target_year+1)] + ['mkt_val_{}'.format(year) for year in range(start_year, target_year+1)] + ['cost_val_{}'.format(year) for year in range(start_year, target_year+1)]
inputs = ['salep_{}'.format(year) for year in range(start_year, target_year)] + ['mkt_val_{}'.format(year) for year in range(start_year, target_year)] + ['cost_val_{}'.format(year) for year in range(start_year, target_year)]
#inputs = ['tract_over_city_{}'.format(year) for year in range(start_year, target_year+1)]
target = 'count_{}'.format(target_year)
#target = 'count_{}'.format(target_year)
columns =  inputs + [target]

###### Run Analysis ######
# kdfcostval = dfs['base_cost_val_mean']
#for year in range(2013, 2018):
#    do_the_thing(df, target, year)
    
results = {}
results_dfs = {}    

#mkt_val_mean = dfs['base_mkt_val_median']

df_input = df[columns]
#results['Lasso'] = fs.getTopFeaturesLasso(df_input, target)
results['Ridge'] = fs.getTopFeaturesRidge(df_input,target)
results['Linear'] = fs.getTopFeaturesLinear(df_input, target)
results['RandomForest'] = fs.getTopFeaturesRandomForest(df_input,target)
results['Correlation'] = fs.getTopFeaturesF(df_input, target)
results['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df_input, target)))
results_dfs = pd.DataFrame(results)

#for df in dfs:
#    df_temp = dfs[df][columns]
#    results[df] = {}
#    print(df)
#    results[df]['Lasso'] = fs.getTopFeaturesLasso(df_temp, target)
#    results[df]['Ridge'] = fs.getTopFeaturesRidge(df_temp,target)
#    results[df]['Linear'] = fs.getTopFeaturesLinear(df_temp, target)
#    results[df]['RandomForest'] = fs.getTopFeaturesRandomForest(df_temp,target)
#    results[df]['Correlation'] = fs.getTopFeaturesF(df_temp, target)
#    results[df]['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df_temp, target)))
#    results_dfs[df] = pd.DataFrame(results[df])

results_dfs_base_cost_val_mean = results_dfs['base_cost_val_mean'] 
results_dfs_base_cost_val_median = results_dfs['base_cost_val_median']
results_dfs_base_mkt_val_mean = results_dfs['base_mkt_val_mean']
results_dfs_base_mkt_val_median = results_dfs['base_mkt_val_median']
results_dfs_base_salep_mean = results_dfs['base_salep_mean']
results_dfs_base_salep_median = results_dfs['base_salep_median']

#results['RandomLasso'] = getTopFeaturesRandomLasso(df,'target')
#results['Lasso'] = fs.getTopFeaturesLasso(df, target)
#results['Ridge'] = fs.getTopFeaturesRidge(df,target)
#results['Linear'] = fs.getTopFeaturesLinear(df, target)
#results['RandomForest'] = fs.getTopFeaturesRandomForest(df,target)
#results['Correlation'] = fs.getTopFeaturesF(df, target)
#results['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df, target)))


methods = list(results.keys())
r = {}