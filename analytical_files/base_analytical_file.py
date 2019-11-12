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

###### Initialize Constants ######
START_YEAR = 2012
END_YEAR = 2018

# Filter specifications on parcel data 
parcels_columns = ['fid', 'year', 'tract', 'salep', 'mkt_val', 'cost_val']
metrics = ['salep', 'mkt_val', 'cost_val']

###### Load Raw Data ######
parcels_raw = pd.read_csv(project_directory+'1_analytical_files/combined/tulsa_parcel_tps_tract_association.csv')
students_raw = pd.read_csv(project_directory+'1_analytical_files/tps_covariates/tps_student_counts_by_tract.csv')


#########################
### Exploratory Stats ###
#########################
parcels = parcels_raw.copy()
## Tabulate Counts of Parcel Types ##
parcel_types = pd.crosstab([parcels['par_type'], parcels['accttype']], parcels['proptype'])

## Tabulate Non-zero metric data ##
parcels['salep_nz'] = np.heaviside(parcels['salep'], 0)
parcels['mkt_val_nz'] = np.heaviside(parcels['mkt_val'], 0)
parcels['cost_val_nz'] = np.heaviside(parcels['cost_val'], 0)
metric_non_zero_tab = pd.crosstab([parcels['mkt_val_nz'], parcels['cost_val_nz']], parcels['salep_nz'])

## Define filters, based on tabulation 
parcels_filter = {'year': range(START_YEAR, END_YEAR+1),
#    'par_type': "PARCEL",
#    'accttype': "Residential", 
#    'proptype': "Residential",
}


############################
###### Transform Data ######
############################
# Apply filter 
parcels = parcels[parcels['year'].isin(list(parcels_filter['year']))]
#parcels = parcels[parcels['mkt_val'] != 0]
#parcels = parcels[parcels['par_type'] == parcels_filter['par_type']]
#parcels = parcels[parcels['accttype'] == parcels_filter['accttype']]
#parcels = parcels[parcels['proptype'] == parcels_filter['proptype']]

# Inspect zero-valued parcels 
zeroed_parcels = parcels[(parcels['mkt_val'] == 0) & (parcels['cost_val'] == 0)]
zeroed_parcels_tab = pd.crosstab([zeroed_parcels['par_type'], zeroed_parcels['accttype']], zeroed_parcels['proptype'], dropna=False)

# Keep columns of interest #
parcels = parcels[parcels_columns]

# Get baseline (City-wide) yearly changes
#city_trend = parcels.groupby(['year']).agg({'fid':['count'], 
#    'salep':['sum', 'mean'], 
#    'mkt_val':['sum', 'mean'], 
#    'cost_val':['sum', 'mean']}).reset_index(level=['year'])


######################################
### Create City-Level Aggregations ###
######################################
parcels['year'] = parcels['year'].astype(int)
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
# Convert year to string #
parcels['year'] = parcels['year'].astype(str)

tract_trends = parcels.groupby(['year', 'tract']).agg({'salep':['mean', 'median'], 
    'mkt_val':['mean', 'median'], 
    'cost_val':['mean', 'median']}).reset_index(level=['year'])

# Rename columns #
tract_trends.columns = ['_'.join(tup).rstrip('_') for tup in tract_trends.columns.values]

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

### Inspect Missingness ###
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