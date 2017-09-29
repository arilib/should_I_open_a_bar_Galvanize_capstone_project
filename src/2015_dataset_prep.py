import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# #load the cleaned csv's from us census
# # all demographic data
# df1 = pd.read_csv('../data/co-est2015-alldata-utf8-3142.csv')
# # all geographic data
# df2 = pd.read_csv('../data/2015_Gaz_counties_national-utf8-3142.csv')
#
# #create new dataframe from df2 only with the column names related 2015
# df1_rc = df1[['STNAME', 'CTYNAME', 'POPESTIMATE2015', 'NPOPCHG_2015',]]
# df2_rc = df2[['USPS', 'GEOID', 'ALAND_SQMI', 'INTPTLAT', 'INTPTLONG']]
#
# # Combine the two dataframes
# comb_df = pd.concat([df1_rc, df2_rc], axis=1,)
#
# Generate new features based on the previous ones
# comb_df['log_aland_sqmi'] = np.log10(comb_df['ALAND_SQMI'])
# comb_df['log_pop_est_2015'] = np.log10(comb_df['POPESTIMATE2015'])
# comb_df['pop_den_2015'] = comb_df['POPESTIMATE2015']/comb_df['ALAND_SQMI']
# comb_df['log_pop_den_2015'] = np.log10(comb_df['pop_den_2015'])
# comb_df['pop_ch_10000_2015'] = 10000*comb_df['NPOPCHG_2015']/comb_df['POPESTIMATE2015']
# comb_df['county_name'] = comb_df['CTYNAME'] + ', ' + comb_df['USP']
#
#
# # Sort the columns
# sequence = ['STNAME', 'USPS', 'GEOID', 'county_name', 'ALAND_SQMI', 'log_aland_sqmi', 'POPESTIMATE2015', 'log_pop_est_2015',
# 'pop_den_2015', 'log_pop_den_2015', 'NPOPCHG_2015', 'pop_ch_10000_2015', 'INTPTLAT', 'INTPTLONG']
# sorted_df = comb_df.reindex(columns=sequence)
#
# # Rename the columns
# sorted_df.columns = ['state_name', 'state_code', 'geo_id', 'county_name', 'area_sqmi', 'log_aland_sqmi', 'pop_est_2015',
#  'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015', 'tot_pop_ch_2015', 'pop_ch_10000_2015', 'lat', 'long']
#
# # Saves the dataframe as a csv as a checkpoint
# sorted_df.to_csv('../data/2015_pop_and_area.csv')
#
# From the notebook
df_pop = pd.read_csv('../data/2015_pop_and_area.csv')
df_pop2 = df_pop[['geo_id', 'state_name', 'state_code', 'county_name', 'area_sqmi', 'log_aland_sqmi', 'pop_est_2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015']]

df_pop2['geo_id'] = df_pop2['geo_id'].astype(str)

dfb1 = processing('../data/BP_2015_00A1_with_ann_bar_utf8.csv', 'ESTAB', 'bar_big')

result = pd.merge(df_pop2, dfb1, on='geo_id', how='outer')

dfb2 = processing('../data/NES_2015_00A2_with_ann_bar_utf8.csv', 'NESTAB', 'bar_small', all_numeric=False)

result2 = pd.merge(result, dfb2, on='geo_id', how='left')

resultD = result2.iloc[np.where(result2["bar_small"] == 'D')]
resultS = result2.iloc[np.where(result2["bar_small"] == 'S')]
resultNDS = result2.iloc[np.where(result2["bar_small"] != 'D') and np.where(result2["bar_small"] != 'S')]

resultD['pop_est_2015'].hist()
plt.show()

resultS['pop_est_2015'].hist()
plt.show()

resultNDS['log_pop_est_2015'].hist()
plt.show()

dfh1 = processing('../data/BP_2015_00A1_with_ann_hot_utf8.csv', 'ESTAB', 'hot_big')
dfh1.head()

dfh2 = processing('../data/NES_2015_00A2_with_ann_hot_utf8.csv', 'NESTAB', 'hot_small', all_numeric=False)
dfh2.head()

result3 = pd.merge(result2, dfh1, on='geo_id', how='left')
result4 = pd.merge(result3, dfh2, on='geo_id', how='left')
result4.head()

result_any_S = result4.loc[(result4['bar_small'] == 'S') | (result4['hot_small'] == 'S')]
result_any_S

result_no_noval = result4.loc[(result4['bar_small'] != 'S') & (result4['hot_small'] != 'S')
                              & (result4['bar_small'] != 'D') & (result4['hot_small'] != 'D')]

result_no_hot_noval = result4.loc[(result4['hot_small'] != 'S')
                              & (result4['hot_small'] != 'D')]
result_no_hot_noval['hot_small'] = result_no_hot_noval['hot_small'].astype(float)
min_hot_noval = result_no_hot_noval['hot_small'].min()
min_hot_noval

result_no_bar_noval = result4.loc[(result4['bar_small'] != 'S')
                              & (result4['bar_small'] != 'D')]
result_no_bar_noval['bar_small'] = result_no_bar_noval['bar_small'].astype(float)
min_bar_noval = result_no_bar_noval['bar_small'].min()
min_bar_noval

result4['hot_big'] = result4['hot_big'].astype(float)
min_hot_noval_big = result4['hot_big'].min()


result4['bar_big'] = result4['bar_big'].astype(float)
min_bar_noval_big = result4['bar_big'].min()
print(min_hot_noval_big, min_bar_noval_big)

result4['bar_big'].fillna(min_bar_noval_big, inplace=True)
result4['hot_big'].fillna(min_hot_noval_big, inplace=True)

result4['bar_small'].fillna(3, inplace=True)
result4['hot_small'].fillna(3, inplace=True)

result4.replace('D', 1.5, inplace=True)
result4.replace('S', 1.5, inplace=True)

result4['hotels'] = result4['hot_big'].astype(float) + result4['hot_small'].astype(float)
result4['bars'] = result4['bar_big'].astype(float) + result4['bar_small'].astype(float)

result4.to_csv('../data/2015_master_sd_1_5_nan_to_min.csv')

result4_toy = result4[['geo_id', 'state_name', 'state_code', 'county_name', 'area_sqmi', 'pop_est_2015', 'hotels', 'bars']]
result4_toy.to_csv('../data/2015_toy_sd_1_5_nan_to_min.csv')


result4_toy['log_bars'] = np.log10(result4_toy['bars'])
result4_toy['log_hotels'] = np.log10(result4_toy['hotels'])

if __name__ == '__main__':
