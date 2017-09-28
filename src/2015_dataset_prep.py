import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Load the cleaned csv's from us census
# This file has the demographic information
df1 = pd.read_csv('../data/co-est2015-alldata-utf8-3142.csv')
# This file has the geographic information
df2 = pd.read_csv('../data/2015_Gaz_counties_national-utf8-3142.csv')

# Create new dataframe from df1 with only the column names related 2015
df1_rc = df1[['STNAME', 'CTYNAME', 'POPESTIMATE2015', 'NPOPCHG_2015',]]
df2_rc = df2[['USPS', 'GEOID', 'ALAND_SQMI', 'INTPTLAT', 'INTPTLONG']]
comb_df = pd.concat([df1_rc, df2_rc], axis=1,)

# Create a few new features from the available ones
comb_df['log_conty_area_sqmi'] = np.log(comb_df['ALAND_SQMI'])
comb_df['log_pop_est_2015'] = np.log(comb_df['POPESTIMATE2015'])
comb_df['pop_den_2015'] = comb_df['POPESTIMATE2015']/comb_df['ALAND_SQMI']
comb_df['log_pop_den_2015'] = np.log(comb_df['pop_den_2015'])
comb_df['pop_ch_10000_2015'] = 10000*comb_df['NPOPCHG_2015']/comb_df['POPESTIMATE2015']


# Resorts the columns
sequence = ['STNAME', 'USPS', 'GEOID', 'CTYNAME', 'ALAND_SQMI', 'log_aland_sqmi', 'POPESTIMATE2015', 'log_pop_est_2015',
'pop_den_2015', 'log_pop_den_2015', 'NPOPCHG_2015', 'pop_ch_10000_2015', 'INTPTLAT', 'INTPTLONG']
sorted_df = comb_df.reindex(columns=sequence)

# Rename the columns
sorted_df.columns = ['state_name', 'state_code', 'geo_id', 'county_name', 'area_sqmi', 'log_aland_sqmi', 'pop_est_2015',
 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015', 'tot_pop_ch_2015', 'pop_ch_10000_2015', 'lat', 'long']

# Saves the dataframe to a csv file
sorted_df.to_csv('../data/2015_pop_and_area.csv')
