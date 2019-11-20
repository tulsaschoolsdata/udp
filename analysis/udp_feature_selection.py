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

###### Initialize ######
# Specify year range (and target year) #
start_year = 2012
target_year = 2018
metrics = ['salep', 'mkt_val', 'cost_val']
dftypes = ['resid', 'comme', 'combo']

# Identify latest batch of analytical files #
os.chdir(project_directory+'1_analytical_files') # go to folder with outputs 
dirs = [d for d in os.listdir('.') if os.path.isdir(d)] # grab list of directories
latest_dir = max(dirs, key=os.path.getmtime) # Identify latest directory 
os.chdir(project_directory)

###### Run Analysis, iteratively ######
results_dfs = {}    
for dftype in dftypes:
    ###### Import Data ######
    df = pd.read_csv(project_directory+'1_analytical_files/{}/base_file_wide_{}.csv'.format(latest_dir, dftype))
    ###### Prep data for analysis ######
    # Keep columns of interest
    inputs = ['salep_{}'.format(year) for year in range(start_year, target_year)] + ['mkt_val_{}'.format(year) for year in range(start_year, target_year)] + ['cost_val_{}'.format(year) for year in range(start_year, target_year)]
    target = 'count_{}'.format(target_year)
    columns =  inputs + [target]    
    df_input = df[columns]
    ###### Run Analysis ######
    results = {}
    #results['Lasso'] = fs.getTopFeaturesLasso(df_input, target)
    results['Ridge'] = fs.getTopFeaturesRidge(df_input,target)
    results['Linear'] = fs.getTopFeaturesLinear(df_input, target)
    results['RandomForest'] = fs.getTopFeaturesRandomForest(df_input,target)
    results['Correlation'] = fs.getTopFeaturesF(df_input, target)
    results['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df_input, target)))
    results_dfs[dftype] = pd.DataFrame(results)

#kresultresid = results_dfs['resid']
#methods = list(results.keys())
#r = {}