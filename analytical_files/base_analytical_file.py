"""
File: base_analytical_file.py
Author: zachandfox
Date Created: 2019-11--1

Description: Compile various datasets into a base analytical file
"""

from CONSTANTS import CURRENT_YEAR
from __tpsdata__ import project_directory, pp
from datetime import datetime as dt
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
parcels_filter = {'year': range(START_YEAR, END_YEAR+1),
#    'par_type': "PARCEL",
#    'accttype': "Residential", 
#    'proptype': "Residential",
}

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
    
    ## Create subsets #
    #parcels_combo = parcels.copy()
    #parcels_resid = parcels.copy()
    #parcels_comme = parcels.copy()
    
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
parcels = parcels_raw.copy()

# Coerce year column to int type
parcels['year'] = parcels['year'].astype(int)

# Create column for 'residential' vs 'commercial'
parcels['is_resid'] = np.any(tuple([parcels[typeid]=='Residential' for typeid in typeids]), axis=0)
parcels['is_comme'] = np.any(tuple([parcels[typeid]=='Commercial' for typeid in typeids]), axis=0)

# Apply year filter 
parcels = parcels[parcels['year'].isin(list(parcels_filter['year']))]

# Create 'residential', 'commercial' dataframes
parcels_resid = parcels[parcels['is_resid']] # Keep if is_residential 
parcels_resid = parcels_resid[parcels_resid['par_type']=='PARCEL']
parcels_comme = parcels[parcels['is_comme']]

if remove_zeroes:
    parcels_resid = parcels_resid[parcels_resid['mkt_val'] != 0]
    parcels_comme = parcels_comme[parcels_comme['mkt_val'] != 0]
#parcels = parcels[parcels['mkt_val'] != 0]
#parcels = parcels[parcels['par_type'] == parcels_filter['par_type']]
#parcels = parcels[parcels['accttype'] == parcels_filter['accttype']]
#parcels = parcels[parcels['proptype'] == parcels_filter['proptype']]

# Keep columns of interest #
parcels_resid = parcels_resid[parcels_columns]
parcels_comme = parcels_comme[parcels_columns]

# Create combo dataframe #
dfs = {'resid': parcels_resid, 'comme':parcels_comme}
parcels_combo = pd.concat(dfs, ignore_index=True)
dfs['combo'] = parcels_combo
# Get baseline (City-wide) yearly changes
#city_trend = parcels.groupby(['year']).agg({'fid':['count'], 
#    'salep':['sum', 'mean'], 
#    'mkt_val':['sum', 'mean'], 
#    'cost_val':['sum', 'mean']}).reset_index(level=['year'])

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


######################################
### Create City-Level Aggregations ###
######################################
trends = {}
trends['city'] = {}
city_trend_list = {}

# Aggregate #
city_trend = parcels.groupby(['year']).agg({'salep':['mean', 'median'], 
    'mkt_val':['mean', 'median'], 
    'cost_val':['mean', 'median']}).reset_index(level=['year'])

# Rename columns #
city_trend.columns = ['_'.join(tup).rstrip('_') for tup in city_trend.columns.values]

# Set year as index #
city_trend = city_trend.set_index('year')

# Split into mean and median, then transpose #
city_trend_mean = city_trend[[metric+'_mean' for metric in metrics]].T
city_trend_median = city_trend[[metric+'_median' for metric in metrics]].T

# Transpose #
#city_trend = city_trend.T

# Create columns for year-by-year change #
for year in range(START_YEAR, END_YEAR):
    city_trend_mean['city_mean_change_'+str(year+1)] = (city_trend_mean[year+1]-city_trend_mean[year])
    city_trend_median['city_median_change_'+str(year+1)] = (city_trend_median[year+1]-city_trend_median[year])

# Drop year columns, Transpose #
city_trend_mean = city_trend_mean.drop(columns=range(START_YEAR, END_YEAR+1))
city_trend_median = city_trend_median.drop(columns=range(START_YEAR, END_YEAR+1))    


#######################################
### Create Tract-level Aggregations ###
#######################################
trends = {}
fulltables = {}

def pop_std(x):
    return x.std(ddof=0)
def pop_mean(x):
    return x.mean(ddof=0)

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
    tract = temp.groupby(['year', 'tract']).agg({'salep':aggs, 
                         'mkt_val':aggs, 
                         'cost_val':aggs}).reset_index(level=['year']).reset_index() 
    tract.columns = ['tract', 'year',
                    'salep_mean_tract',
                    'mkt_val_mean_tract',
                    'cost_val_mean_tract',]
    citytract = pd.merge(tract, city, how='left', on=['year'], suffixes=('_tract', '_city')) # Merge together state and local data
    
    # Calculate normalized tract performance
    for metric in metrics:
        citytract[metric] = (citytract[metric+'_mean_tract']-citytract[metric+'_mean_city'])/citytract[metric+'_std_city']
    fulltables[dftype] = citytract.copy()
    output = citytract[['tract', 'year']+metrics] 
    preshape = output.copy()
    output = output.pivot(index='tract', columns='year')
    output = output.reset_index()
    colnames = ['tract'] + ['salep_{}'.format(year) for year in range(2012,2019)] + ['mkt_val_{}'.format(year) for year in range(2012,2019)] + ['cost_val_{}'.format(year) for year in range(2012,2019)]
    output.columns = colnames
    output = output.dropna()
    # Reshape wider on year
    output.to_csv(project_directory+'1_analytical_files/base_file.csv')
    

    # Aggregate
#    output = temp.groupby(['year', 'tract']).agg({'mkt_val':aggs, 
#                         'mkt_val':aggs, 
#                         'cost_val':aggs}).reset_index(level=['year']) 
    # Rename Columns
#    output.columns = ['_'.join(tup).rstrip('_') for tup in output.columns.values]
#    output = output.reset_index()
#    output = output.pivot(index='tract', columns='year')
#    output[']
    trends['tract'][dftype] = output.copy()
#ktrendtractresid = trends['tract']['resid']
# ktrendcity = trends['city']
# Rename columns #
    
#tract_trends.columns = ['_'.join(tup).rstrip('_') for tup in tract_trends.columns.values]

# Add year to index #
tract_trends = tract_trends.set_index(['year'], append=True)

# Split into mean and median, then transpose #
tract_trends_mean = tract_trends[[metric+'_mean' for metric in metrics]]
tract_trends_median = tract_trends[[metric+'_median' for metric in metrics]]

# Reshape wide on year, transpose #
tract_trends_mean = tract_trends_mean.reset_index()
tract_trends_mean = tract_trends_mean.pivot(index='year', columns='tract').T
#tract_trends_mean = tract_trends_mean.reset_index(level='tract')

tract_trends_median = tract_trends_median.reset_index()
tract_trends_median = tract_trends_median.pivot(index='year', columns='tract').T
#tract_trends_median = tract_trends_median.reset_index(level='tract')


# Rename columns again #
#tract_trends_mean.columns = ['_'.join(tup).rstrip('_') for tup in tract_trends_mean.columns.values]
#tract_trends_median.columns = ['_'.join(tup).rstrip('_') for tup in tract_trends_median.columns.values]
### Inspect aggregations ###
#ktractresid = trends['tract']['resid']

# Missingness #
tract_mean_nulls = tract_trends_mean.isnull()
tract_median_nulls = tract_trends_median.isnull()
tract_median_nulls = tract_median_nulls .groupby(level=[0,1]).sum().astype(int)
tract_mean_nulls = tract_mean_nulls .groupby(level=[0,1]).sum().astype(int)

## Drop tracts w/ missing data (Drops 22 Tracts) #
tract_trends_mean = tract_trends_mean.dropna()
tract_trends_median = tract_trends_median.dropna()

# Transpose

# Create columns for year-by-year change #
for year in range(START_YEAR, END_YEAR):
    tract_trends_mean['tract_mean_change_'+str(year+1)] = (tract_trends_mean[str(year+1)]-tract_trends_mean[str(year)])
    tract_trends_median['tract_median_change_'+str(year+1)] = (tract_trends_median[str(year+1)]-tract_trends_median[str(year)])

# Drop year columns, transpose #
tract_trends_mean = tract_trends_mean.drop(columns=map(str,range(START_YEAR, END_YEAR+1)))
tract_trends_median = tract_trends_median.drop(columns=map(str,range(START_YEAR, END_YEAR+1)))
    
# Drop Tract from Index 
tract_trends_mean = tract_trends_mean.reset_index(level='tract')
tract_trends_median = tract_trends_median.reset_index(level='tract')
## Reshape wide #
##student_counts_wide = student_counts.pivot(index='tract', columns='year', values='count').fillna(0).astype('int64')
##city_trend = city_trend.pivot(index='')
#
## Get tract-level yearly changes 
#tract_trends = parcels.groupby(['year', 'tract']).agg({'acct_num':['count'], 
#    'salep':['sum', 'mean'], 
#    'mkt_val':['sum', 'mean'], 
#    'cost_val':['sum', 'mean']}).reset_index(level=['year'])


###############################################
### Compare Tract trends against City Trend ###
###############################################
# Merge
tract_v_city_mean = tract_trends_mean.merge(city_trend_mean, how='left', left_index=True, right_index=True)
tract_v_city_median = tract_trends_median.merge(city_trend_median, how='left', left_index=True, right_index=True)

# Relative to baseline w/in year #
for year in range(START_YEAR+1, END_YEAR+1):
    tract_v_city_mean['tract_over_city_'+str(year)] = tract_v_city_mean['tract_mean_change_{}'.format(year)] - tract_v_city_mean['city_mean_change_{}'.format(year)]
    tract_v_city_median['tract_over_city_'+str(year)] = tract_v_city_median['tract_median_change_{}'.format(year)] - tract_v_city_median['city_median_change_{}'.format(year)]
    
# Add Tract to index
tract_v_city_mean = tract_v_city_mean.set_index(['tract'], append=True)
tract_v_city_median = tract_v_city_median.set_index(['tract'], append=True)

# Keep relevant columns #
tract_v_city_mean = tract_v_city_mean[['tract_over_city_{}'.format(year) for year in range(START_YEAR+1, END_YEAR+1)]]
tract_v_city_median = tract_v_city_median[['tract_over_city_{}'.format(year) for year in range(START_YEAR+1, END_YEAR+1)]]

# Move Metric out of index #
tract_v_city_mean = tract_v_city_mean.reset_index(level=0).rename(columns={'level_0': 'metric'})
tract_v_city_median = tract_v_city_median.reset_index(level=0).rename(columns={'level_0': 'metric'})

# Pivot Wide #
tract_v_city_mean = tract_v_city_mean.reset_index()
tract_v_city_mean = tract_v_city_mean.pivot(index='tract', columns='metric')
tract_v_city_median = tract_v_city_median.reset_index()
tract_v_city_median = tract_v_city_median.pivot(index='tract', columns='metric')

# Set tract as index 
#tract_v_city_mean = tract_v_city_mean.set_index(('tract',''))
#tract_v_city_mean.index.names=['tract']

## Split by metric ##
salep_mean = tract_v_city_mean.iloc[:, tract_v_city_mean.columns.get_level_values(1)=='salep_mean']
mkt_val_mean = tract_v_city_mean.iloc[:, tract_v_city_mean.columns.get_level_values(1)=='mkt_val_mean']
cost_val_mean = tract_v_city_mean.iloc[:, tract_v_city_mean.columns.get_level_values(1)=='cost_val_mean']
salep_median = tract_v_city_median.iloc[:, tract_v_city_median.columns.get_level_values(1)=='salep_median']
mkt_val_median = tract_v_city_median.iloc[:, tract_v_city_median.columns.get_level_values(1)=='mkt_val_median']
cost_val_median = tract_v_city_median.iloc[:, tract_v_city_median.columns.get_level_values(1)=='cost_val_median']

## Clean columns ##
salep_mean.columns = salep_mean.columns.get_level_values(0)
mkt_val_mean.columns = mkt_val_mean.columns.get_level_values(0)
cost_val_mean.columns = cost_val_mean.columns.get_level_values(0)
salep_median.columns = salep_median.columns.get_level_values(0)
mkt_val_median.columns = mkt_val_median.columns.get_level_values(0)
cost_val_median.columns = cost_val_median.columns.get_level_values(0)


##############################
### Transform Student Data ###
##############################
students = students_raw.copy()
### Rename columns in student counts ###
student_columns = list(students.columns)
# Ignore first column ('tract')
student_columns.pop(0)
# Store new names #
new_names = [(i, 'count_' + i) for i in student_columns]
# Rename columns #
students.rename(columns = dict(new_names), inplace=True)
# Add tract to index 
students = students.set_index(['tract'])


###########################
###### Combine Data #######
###########################
output2 = students.merge(output, on='tract')
output2.to_csv(project_directory+'1_analytical_files/base_file.csv')


### Inspect Nulls ###
inspect_null_mean = students.merge(tract_mean_nulls, on='tract')

# Inner join both datasets
base_salep_mean = students.merge(salep_mean, on='tract')
base_mkt_val_mean = students.merge(mkt_val_mean, on='tract')
base_cost_val_mean = students.merge(cost_val_mean, on='tract')
base_salep_median = students.merge(salep_median, on='tract')
base_mkt_val_median = students.merge(mkt_val_median, on='tract')
base_cost_val_median = students.merge(cost_val_median, on='tract')


#########################
###### Export Data ######
#########################
base_salep_mean.to_csv(project_directory+'1_analytical_files/base_salep_mean.csv')
base_mkt_val_mean.to_csv(project_directory+'1_analytical_files/base_mkt_val_mean.csv')
base_cost_val_mean.to_csv(project_directory+'1_analytical_files/base_cost_val_mean.csv')
base_salep_median.to_csv(project_directory+'1_analytical_files/base_salep_median.csv')
base_mkt_val_median.to_csv(project_directory+'1_analytical_files/base_mkt_val_median.csv')
base_cost_val_median.to_csv(project_directory+'1_analytical_files/base_cost_val_median.csv')