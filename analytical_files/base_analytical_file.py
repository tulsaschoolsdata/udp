"""
File: base_analytical_file.py
Author: zachandfox
Date Created: 2019-11--1

Description: Compile various datasets into a base analytical file
"""

from CONSTANTS import CURRENT_YEAR
from __tpsdata__ import project_directory, pp
from datetime import datetime as dt
import os
import pandas as pd
import numpy as np
import seaborn as sns

# TODO: Undo year-by-year change for <metrics>
# TODO: search for outliers in underlying data: if not residential, remove

# TODO: Create residential, commercial, combined equivalents
# TODO: Remove outliers
# TODO: Shape: count_2010-count_2020, tract_v_city_{metrics}
# TODO: tract_v_city_{metrics}:
#    standard deviations of tract against city
    

###### Initialize Constants ######
START_YEAR = 2012
END_YEAR = 2018

# Define useful lists
parcels_columns = ['fid', 'year', 'tract', 'salep', 'mkt_val', 'cost_val']
metrics = ['salep', 'mkt_val', 'cost_val']
typeids = ['accttype', 'proptype']
dftypes = ['resid', 'comme', 'combo']

# Toggles:
remove_zeroes = False # Remove zero-valued entries for metrics
explore_stats = False
medians = False
if medians:
    aggs = ['mean', 'median']
else:
    aggs = ['mean']


## Define filters, based on tabulation 
parcels_filter = {'year': range(START_YEAR, END_YEAR+1)}

###### Load Raw Data ######
parcels_raw = pd.read_csv(project_directory+'1_analytical_files/combined/tulsa_parcel_tps_tract_association.csv')
students_raw = pd.read_csv(project_directory+'1_analytical_files/tps_covariates/tps_student_counts_by_tract.csv')


#########################
### Exploratory Stats ###
#########################
if explore_stats:
    parcels = parcels_raw.copy()
    ## Tabulate Counts of Parcel Types ##
    parcel_types = pd.crosstab([parcels['par_type'], parcels['accttype']], parcels['proptype'])
    
    ## Tabulate Non-zero metric data ##
    parcels['salep_nz'] = np.heaviside(parcels['salep'], 0)
    parcels['mkt_val_nz'] = np.heaviside(parcels['mkt_val'], 0)
    parcels['cost_val_nz'] = np.heaviside(parcels['cost_val'], 0)
    metric_non_zero_tab = pd.crosstab([parcels['mkt_val_nz'], parcels['cost_val_nz']], parcels['salep_nz'])
    
    # Inspect zero-valued parcels 
    zeroed_parcels = parcels[(parcels['mkt_val'] == 0) & (parcels['cost_val'] == 0)]
    zeroed_parcels_tab = pd.crosstab([zeroed_parcels['par_type'], zeroed_parcels['accttype']], zeroed_parcels['proptype'], dropna=False)
    

############################
###### Transform Data ######
############################
### Students ##################################################################
students = students_raw.copy()
# Rename columns in student counts #
student_columns = list(students.columns)
# Ignore first column ('tract')
student_columns.pop(0)
# Store new names #
new_names = [(i, 'count_' + i) for i in student_columns]
# Rename columns #
students.rename(columns = dict(new_names), inplace=True)
# Add tract to index 
students = students.set_index(['tract'])

    
### Parcels ###################################################################
parcels = parcels_raw.copy()

# Coerce year column to int type
parcels['year'] = parcels['year'].astype(int)

# Create column for 'residential' vs 'commercial'
parcels['is_resid'] = (parcels['proptype']=='Condo') | (parcels['accttype']=='Residential')
parcels['is_comme'] = np.any(tuple([parcels[typeid]=='Commercial' for typeid in typeids]), axis=0)

# Apply year filter 
parcels = parcels[parcels['year'].isin(list(parcels_filter['year']))]

# Create 'residential', 'commercial' dataframes
#parcels_resid = parcels[parcels['is_resid']] # Keep if is_residential 
parcels_resid = parcels[parcels['is_resid']]
#parcels_resid = parcels_resid[parcels_resid['par_type']=='PARCEL']
parcels_comme = parcels[parcels['is_comme']]

# Inspect parcel types #
resid_types = pd.crosstab([parcels_resid['par_type'], parcels_resid['accttype']], 
                          parcels_resid['proptype'])

# Keep columns of interest #
parcels_resid = parcels_resid[parcels_columns]
parcels_comme = parcels_comme[parcels_columns]

# Create combo dataframe #
dfs = {'resid': parcels_resid, 'comme':parcels_comme}
parcels_combo = pd.concat(dfs, ignore_index=True)
dfs['combo'] = parcels_combo

### Exploratory Stats ###
dfs_stats = {}
for dftype in dftypes:
    dfs_stats[dftype] = {}
    temp = dfs[dftype].copy()
    temp = temp[['tract', 'year', 'salep', 'mkt_val', 'cost_val']].set_index(['tract', 'year']) # Restructure
    pairplot = sns.pairplot(temp) # Create pairplot distributions 
    pairplot.fig.suptitle(dftype, y=1.08) # Set title
    dfs_stats[dftype]['pairplot'] = pairplot # Assign to stats dictionary
    for metric in metrics:
        dfs_stats[dftype][metric] = sns.distplot(temp[metric]) # Create univariate distributions 
# kresid = dfs['resid'].copy()
# TODO: use programmatic filtering on Sdev's from mean
metric_filters = {}
metric_filters['resid'] = {}
metric_filters['resid']['salep'] = {}
metric_filters['resid']['salep']['min'] = 0
metric_filters['resid']['salep']['max'] = 5e6
metric_filters['resid']['mkt_val'] = {}
metric_filters['resid']['mkt_val']['min'] = 0
metric_filters['resid']['mkt_val']['max'] = 3e6
metric_filters['resid']['cost_val'] = {}
metric_filters['resid']['cost_val']['min'] = 0
metric_filters['resid']['cost_val']['max'] = 5e6
# kcomme = dfs['comme'].copy()
metric_filters['comme'] = {}
metric_filters['comme']['salep'] = {}
metric_filters['comme']['salep']['min'] = 0
metric_filters['comme']['salep']['max'] = 1e8
metric_filters['comme']['mkt_val'] = {}
metric_filters['comme']['mkt_val']['min'] = 0
metric_filters['comme']['mkt_val']['max'] = 5e7
metric_filters['comme']['cost_val'] = {}
metric_filters['comme']['cost_val']['min'] = 0
metric_filters['comme']['cost_val']['max'] = 1e8

# Apply filters #
dfs_filtered = {}
for dftype in ['resid', 'comme']:
    temp = dfs[dftype].copy()
    for metric in ['salep', 'mkt_val']:
        temp_min = temp[metric] > metric_filters[dftype][metric]['min'] # flag if obs is above min
        temp_max = temp[metric] < metric_filters[dftype][metric]['max'] # flag if obs is below max
        temp = temp[temp_min & temp_max]
    dfs_filtered[dftype] = temp.copy()
dfs_filtered['combo'] = pd.concat(dfs_filtered, ignore_index=True)
    
### Exploratory Stats ###
dfs_stats_filtered = {}
for dftype in dftypes:
    dfs_stats_filtered[dftype] = {}
    temp = dfs_filtered[dftype].copy()
    temp = temp[['tract', 'year', 'salep', 'mkt_val', 'cost_val']].set_index(['tract', 'year']) # Restructure
    pairplot = sns.pairplot(temp) # Create pairplot distributions 
    pairplot.fig.suptitle(dftype, y=1.08) # Set title
    dfs_stats_filtered[dftype]['pairplot'] = pairplot # Assign to stats dictionary
    for metric in metrics:
        dfs_stats_filtered[dftype][metric] = sns.distplot(temp[metric]) # Create univariate distributions 
# kresidf = dfs_filtered['resid']
# kcommef = dfs_filtered['comme']

### Commit filter ###
dfs = dfs_filtered

### TODO: Case-study the removed properties ###

#######################################
### Create Tract-level Aggregations ###
#######################################
tract_long = {}
tract_wide = {}

def pop_std(x):
    return x.std(ddof=0)

# Aggregate #
for dftype in dftypes:
    temp = dfs[dftype].copy()
    temp['year'] = temp['year'].astype(str) # Coerce year to string
    # City-level aggregation 
    city = temp.groupby(['year']).agg({metric: aggs+[pop_std] for metric in metrics})
    city = city.reset_index()
    city.columns = ['year', 
                    'salep_mean_city', 'salep_std_city', 
                    'mkt_val_mean_city', 'mkt_val_std_city', 
                    'cost_val_mean_city', 'cost_val_std_city']
    # Tract-Level aggregation #
    tract = temp.groupby(['year', 'tract']).agg({'salep':aggs, 
                         'mkt_val':aggs, 
                         'cost_val':aggs}).reset_index(level=['year']).reset_index() 
    tract.columns = ['tract', 'year',
                    'salep_mean_tract',
                    'mkt_val_mean_tract',
                    'cost_val_mean_tract',]
    # Combine City and Tract
    citytract = pd.merge(tract, city, how='left', on=['year'], suffixes=('_tract', '_city')) # Merge together state and local data
    # Calculate normalized tract performance
    for metric in metrics:
        citytract[metric] = (citytract[metric+'_mean_tract']-citytract[metric+'_mean_city'])/citytract[metric+'_std_city']
    # Save, export #
    tract_long[dftype] = citytract.copy()
    # Create wide file for analytical purposes #
    output = citytract[['tract', 'year']+metrics] 
    preshape = output.copy()
    output = output.pivot(index='tract', columns='year')
    output = output.reset_index()
    colnames_ = [['salep_{}'.format(year), 'mkt_val_{}'.format(year), 'cost_val_{}'.format(year)] for year in range(START_YEAR, END_YEAR+1)]
    colnames_ = [item for sublist in colnames_ for item in sublist] # flatten colnames_
    colnames_.sort()
    colnames = ['tract'] + colnames_
    output.columns = colnames
    # Impute values for NA's
    output = output.dropna()
    tract_wide[dftype] = output


###########################
###### Combine Data #######
###########################
df_out_wide = {}
df_out_long = {}

for dftype in dftypes:
    df_out_wide[dftype] = pd.merge(students, tract_wide[dftype], on='tract')
    df_out_long[dftype] = pd.merge(students, tract_long[dftype], on='tract')


#########################
###### Export Data ######
#########################
now = dt.now()
timestamp = "{}-{}-{}_{}-{}".format(now.year, now.month, now.day, now.hour, now.minute)
os.mkdir(project_directory+'1_analytical_files/'+timestamp)
for dftype in dftypes:
    df_out_wide[dftype].to_csv(project_directory+'1_analytical_files/{}/base_file_wide_{}.csv'.format(dftype))
    df_out_long[dftype].to_csv(project_directory+'1_analytical_files/{}/base_file_long_{}.csv'.format(dftype))