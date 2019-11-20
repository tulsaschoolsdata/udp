"""
File: tulsa_parcels.py
Author: zachandfox
Date Created: 2019-11-01

Description: Pull in raw parcel data, clean it, export it

"""

from __tpsdata__ import project_directory
from CONSTANTS import START_YEAR
import pandas as pd

###### Initialize Constants ######
# Define columns to keep #
columns = ['FID', 'AREA', 'PAR_TYPE', 'LON', 'LAT', 'PIC_YEAR', 'ACCTTYPE', 'SECTION', 'SALEP', 'CONFIRMUN', 'VALIDINV', 'VAC_SALE', 'QUALITY', 'PROPTYPE', 'CONDITION', 'LEADESC', 'MKT_VAL', 'COST_VAL', 'Neighborhood']

###### Load Raw Data #######
tulsa_parcels = pd.read_csv(project_directory+'0_raw_files/TulsaParcelsNeighborhood.csv')

###### Transform Data ######
# Keep columns of interest #
tulsa_parcels = tulsa_parcels[columns]

# Filter on year > 2009
tulsa_parcels = tulsa_parcels[tulsa_parcels.PIC_YEAR > START_YEAR]

# Rename columns 
tulsa_parcels = tulsa_parcels.rename(columns={'PIC_YEAR': 'year'})

###### Export ######
tulsa_parcels.to_csv(project_directory+'1_analytical_files/tulsa_data/tulsa_parcels.csv')
