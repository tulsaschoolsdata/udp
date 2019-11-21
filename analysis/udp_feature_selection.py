"""
analysis/udp_feature_selection.py
author: zachandfox
date created: 2019-11-12
"""

import analysis.feature_selection_methods as fs
#from CONSTANTS import CURRENT_YEAR
from __tpsdata__ import project_directory
import os
import pandas as pd
#import numpy as np

###### Initialize ######
# Specify year range (and target year) #
start_year = 2012
end_year = 2018
metrics = ['salep', 'mkt_val', 'cost_val']
dftypes = ['resid', 'comme', 'combo']

# Identify latest batch of analytical files #
os.chdir(project_directory+'1_analytical_files') # go to folder with outputs 
dirs = [d for d in os.listdir('.') if os.path.isdir(d)] # grab list of directories
latest_dir = max(dirs, key=os.path.getmtime) # Identify latest directory 
os.chdir(project_directory)

###### Run Analysis, iteratively ######
results_by_year = {}    
for dftype in dftypes:
    ###### Import Data ######
    df_base = pd.read_csv(project_directory+'1_analytical_files/{}/base_file_wide_{}.csv'.format(latest_dir, dftype))
    results_by_year[dftype] = {}
    for target_year in range(start_year+1, end_year+1):
        print(dftype, target_year)
        df = df_base.copy()        
        ###### Prep data for analysis ######
        # Keep columns of interest
        inputs = ['salep_{}'.format(year) for year in range(start_year, target_year)] + ['mkt_val_{}'.format(year) for year in range(start_year, target_year)] + ['cost_val_{}'.format(year) for year in range(start_year, target_year)]
        target = 'count_{}'.format(target_year)
        columns =  inputs + [target]    
        df_input = df[columns]
        ###### Run Analysis ######
        results = {}
        #results['Lasso'] = fs.getTopFeaturesLasso(df_input, target)
        results['rdg'] = fs.getTopFeaturesRidge(df_input,target)
        results['lin'] = fs.getTopFeaturesLinear(df_input, target)
        results['rf'] = fs.getTopFeaturesRandomForest(df_input,target)
        results['cor'] = fs.getTopFeaturesF(df_input, target)
        #results['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df_input, target)))
        dfout = pd.DataFrame(results)
        dfout.rename(columns = {name: name+'{}'.format(target_year) for name in dfout.columns}, inplace=True)
        dfout = dfout.reset_index()
        dfout.rename(columns = {'index':'metric'}, inplace=True)
        results_by_year[dftype][target_year] = dfout

###### Compile Results ######
results_out = {}
for dftype in dftypes:
    temp = results_by_year[dftype][start_year+1]
    for target_year in range(start_year+2, end_year+1):
        print(dftype, target_year)
        temp = pd.merge(temp, results_by_year[dftype][target_year], on='metric', how='outer')
    results_out[dftype] = temp.copy()

koutresid = results_out['resid']
koutcomme = results_out['comme']
koutcombo = results_out['combo']

###### Export Results ######
#outpath = project_directory+'2_output_deliverables/Analysis/'+latest_dir
#os.mkdir(outpath)
for dftype in dftypes:
    results_out[dftype].to_csv(outpath+'/analysis_{}.csv'.format(dftype))

#methods = list(results.keys())
#r = {}
