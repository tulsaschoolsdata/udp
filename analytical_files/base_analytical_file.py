"""
File: base_analytical_file.py
Author: zachandfox
Date Created: 2019-11--1

Description: Compile various datasets into a base analytical file
"""

from __tpsdata__ import project_directory
import pandas as pd
import numpy as np

###### Load Raw Data ######
parcels = pd.read_csv(project_directory+'1_analytical_files/combined/tulsa_parcel_tps_tract_association.csv')
students = pd.read_csv(project_directory+'1_analytical_files/tps_covariates/tps_student_counts_by_tract.csv')


###### Transform Data ######
### TODO: Aggregate Parcel Data by tract ###
# Relative to baseline w/in year #
# Relative to self from last year #

### Rename columns in student counts ###
student_columns = list(students.columns)
# Ignore first column ('tract')
student_columns.pop(0)
# Store new names #
new_names = [(i, 'count_' + i) for i in student_columns]
# Rename columns #
students.rename(columns = dict(new_names), inplace=True)


###### Combine Data #######
# Inner join both datasets
#base_analytical_file = students.merge(parcels, on='tract')


###### Export Data ######
#base_analytical_file.to_csv(project_directory+'1_analytical_files/base_analytical_file.csv')
