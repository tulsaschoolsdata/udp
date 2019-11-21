"""
analysis/udp_feature_selection.py
author: zachandfox
date created: 2019-11-12
"""

import analysis.feature_selection_methods as fs
#from CONSTANTS import CURRENT_YEAR
from __tpsdata__ import REPO_PATH, project_directory
import os
import pandas as pd
#import numpy as np

###### Initialize ######
# Specify year range (and target year) #
start_year = 2012
end_year = 2018
metrics = ['salep', 'mkt_val', 'cost_val']
#dftypes = ['resid', 'comme', 'combo']
dftypes = ['resid']

# Identify latest batch of analytical files #
os.chdir(project_directory+'1_analytical_files') # go to folder with outputs 
dirs = [d for d in os.listdir('.') if os.path.isdir(d)] # grab list of directories
latest_dir = max(dirs, key=os.path.getmtime) # Identify latest directory 
os.chdir(REPO_PATH)

inputs = ['salep_city', 'salep', 'salep_norm', 
          'mkt_val_city', 'mkt_val', 'mkt_val_norm',
          'cost_val_city', 'cost_val', 'cost_val_norm',
          'diff_salep_0', 'diff_salep_1', 'diff_salep_2', 'diff_salep_3', 'diff_salep_4', 'diff_salep_5',
          'diff_salep_norm_0', 'diff_salep_norm_1', 'diff_salep_norm_2', 'diff_salep_norm_3', 'diff_salep_norm_4', 'diff_salep_norm_5',
          'diff_mkt_val_0', 'diff_mkt_val_1', 'diff_mkt_val_2', 'diff_mkt_val_3', 'diff_mkt_val_4', 'diff_mkt_val_5',
          'diff_mkt_val_norm_0', 'diff_mkt_val_norm_1', 'diff_mkt_val_norm_2', 'diff_mkt_val_norm_3', 'diff_mkt_val_norm_4', 'diff_mkt_val_norm_5', 
          'diff_cost_val_0', 'diff_cost_val_1', 'diff_cost_val_2', 'diff_cost_val_3', 'diff_cost_val_4', 'diff_cost_val_5', 
          'diff_cost_val_norm_0', 'diff_cost_val_norm_1', 'diff_cost_val_norm_2', 'diff_cost_val_norm_3', 'diff_cost_val_norm_4', 'diff_cost_val_norm_5']
target = 'count'
#target = 'diff_count_0'

###### Run Analysis, iteratively ######
#results_by_year = {}    
#for dftype in dftypes:
    ###### Import Data ######
#    df_base = pd.read_csv(project_directory+'1_analytical_files/{}/base_file_wide_{}.csv'.format(latest_dir, dftype))
df_base = pd.read_csv(project_directory+'1_analytical_files/{}/base_file_long_{}.csv'.format(latest_dir, dftype))
#    for target_year in range(start_year+1, end_year+1):
#print(dftype, target_year)
df = df_base.copy()        
###### Prep data for analysis ######
# Keep columns of interest
columns =  inputs + [target]    
df_input = df[columns]

###### Run Analysis ######
results = {}
#results['Lasso'] = fs.getTopFeaturesLasso(df_input, target)
#results['rdg'] = fs.getTopFeaturesRidge(df_input,target)
results['lin'] = fs.getTopFeaturesLinear(df_input, target)
#results['rf'] = fs.getTopFeaturesRandomForest(df_input,target)
#results['cor'] = fs.getTopFeaturesF(df_input, target)
results['F'] = fs.getTopFeaturesF(df_input, target)[0]
results['p'] = fs.getTopFeaturesF(df_input, target)[1]
klin = results['lin']
#kF = results['F']
#krf = results['rf']
#results['RFE'] = dict(zip(inputs, fs.getTopFeaturesRFE(df_input, target)))
dfout = pd.DataFrame(results)
#dfout.rename(columns = {name: name+'{}'.format(target_year) for name in dfout.columns}, inplace=True)
#dfout = dfout.reset_index()
#dfout.rename(columns = {'index':'metric'}, inplace=True)
#    results_by_year[dftype][target_year] = dfout

###### Compile Results ######
#results_out = {}
#for dftype in dftypes:
#    temp = results_by_year[dftype][start_year+1]
#    for target_year in range(start_year+2, end_year+1):
#        print(dftype, target_year)
#        temp = pd.merge(temp, results_by_year[dftype][target_year], on='metric', how='outer')
#    results_out[dftype] = temp.copy()
#
#koutresid = results_out['resid'].set_index('metric')
#koutresid = koutresid.reindex(sorted(koutresid.columns), axis=1)
#koutcomme = results_out['comme']
#koutcombo = results_out['combo']
df_plot_out = df_base[['tract', 'year', 'count', 'mkt_val_city', 'mkt_val', 'mkt_val_norm', 'diff_mkt_val_norm_5',
                       'diff_count_0', 'diff_count_1', 'diff_count_2', 'diff_count_3', 'diff_count_4', 'diff_count_5', 
                       'diff_mkt_val_0', 'diff_mkt_val_1', 'diff_mkt_val_2', 'diff_mkt_val_3', 'diff_mkt_val_4', 'diff_mkt_val_5',
                       'diff_mkt_val_norm_0', 'diff_mkt_val_norm_1', 'diff_mkt_val_norm_2', 'diff_mkt_val_norm_3', 'diff_mkt_val_norm_4' ]]
df_plot_out = df_plot_out[df_plot_out['year']==2018]

###### Export Results ######
outpath = project_directory+'2_output_deliverables/Analysis/'+latest_dir
os.mkdir(outpath)
#for dftype in dftypes:
dfout.to_csv(outpath+'/analysis_out.csv'.format(dftype))
df_plot_out.to_csv(outpath+'/plot_out.csv')

#methods = list(results.keys())
#r = {}

