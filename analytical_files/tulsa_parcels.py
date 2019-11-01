"""
File: tulsa_parcels.py
Author: zachandfox
Date Created: 2019-11-01

Description: Pull in raw parcel data, clean it, export it

"""

from __tpsdata__ import project_directory
import numpy as np
import pandas as pd

###### Initialize Constants ######
START_YEAR = 2009
# Define columns to keep #
columns = ['FID', 'AREA', 'PAR_TYPE', 'LON', 'LAT', 'PIC_YEAR', 'ACCTTYPE', 'SECTION', 'SALEP', 'CONFIRMUN', 'VALIDINV', 'VAC_SALE', 'QUALITY', 'PROPTYPE', 'CONDITION', 'LEADESC', 'MKT_VAL', 'COST_VAL', 'Neighborhood']

###### Load Raw Data #######
cot = pd.read_csv(project_directory+'0_raw_files/TulsaParcelsNeighborhood.csv')

###### Transform Data ######
# Filter on year > 2009
cot = cot[cot.PIC_YEAR > START_YEAR]

###### Export ######
cot.to_csv(project_directory+'1_analytical_files/tulsa_data/tulsa_parcels.csv')
