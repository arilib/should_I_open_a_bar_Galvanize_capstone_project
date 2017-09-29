import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def selec_col_df(df, cols):
    rc_df = df[cols]
    return(rc_df)

def combine_dfs(df1, df2):
    joined_df = pd.concat([df1, df2], axis=1,)
    return joined_df


if __name__ == '__main__':
    #load the cleaned csv's from us census
    # all demographic data
    demo_df = pd.read_csv('../data/co-est2015-alldata-utf8-3142.csv')
    # all geographic data
    geo_df = pd.read_csv('../data/2015_Gaz_counties_national-utf8-3142.csv')

    # Create new demo_df with only 2015 columns (the col used were selected by exploring the csv file in a notebook)
    demo_df2 = selec_col_df(demo_df, ['STNAME', 'CTYNAME', 'POPESTIMATE2015', 'NPOPCHG_2015'])

    # Create new geo_df2 with some of the columns of geo_df1 (the col used were selected by exploring the csv file in a notebook)
    geo_df2 = selec_col_df(geo_df, ['USPS', 'GEOID', 'ALAND_SQMI'])

    # Combine the two dataframes side by side
    demo_geo_df = combine_dfs(geo_df2, demo_df2)

    # Generate new features based on the previous ones
    demo_geo_df['log_aland_sqmi'] = np.log10(demo_geo_df['ALAND_SQMI'])
    demo_geo_df['log_pop_est_2015'] = np.log10(demo_geo_df['POPESTIMATE2015'])
    demo_geo_df['pop_den_2015'] = demo_geo_df['POPESTIMATE2015']/demo_geo_df['ALAND_SQMI']
    demo_geo_df['log_pop_den_2015'] = np.log10(demo_geo_df['pop_den_2015'])
    demo_geo_df['pop_ch_10000_2015'] = 10000*demo_geo_df['NPOPCHG_2015']/demo_geo_df['POPESTIMATE2015']
    demo_geo_df['county_name'] = demo_geo_df['CTYNAME'] + ', ' + demo_geo_df['USPS']

    print
    # Sort the columns
    sequence = ['STNAME', 'USPS', 'GEOID', 'county_name', 'ALAND_SQMI', 'log_aland_sqmi', 'POPESTIMATE2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015', 'NPOPCHG_2015', 'pop_ch_10000_2015', 'INTPTLAT', 'INTPTLONG']
    sorted_df = demo_geo_df.reindex(columns = sequence)

    # Rename the columns
    sorted_df.columns = ['state_name', 'state_code', 'geo_id', 'county_name', 'area_sqmi', 'log_aland_sqmi', 'pop_est_2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015', 'tot_pop_ch_2015', 'pop_ch_10000_2015', 'lat', 'long']

    # Saves the dataframe as a csv with all cols as a checkpoint
    demo_geo_df.to_csv('../data/2015_demo_geo.csv')

    # df_pop2 = df_pop[['geo_id', 'state_name', 'state_code', 'county_name', 'area_sqmi', 'log_aland_sqmi', 'pop_est_2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015']]
