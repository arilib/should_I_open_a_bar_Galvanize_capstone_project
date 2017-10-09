import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def processing(filename, col_name_orig, new_col_name, all_numeric=True):
    df1 = pd.read_csv(filename)
    df2 = df1[['GEO.id2', col_name_orig]]
    df2 = df2[df2['GEO.id2'] != 'Id2']
    df2.columns = ['geo_id', new_col_name]
    if all_numeric: df2[new_col_name] = df2[new_col_name].astype(int)
    return(df2)

def selec_col_df(df, cols):
    rc_df = df[cols]
    return(rc_df)

def combine_dfs(df1, df2, left, right, how='left'):
    joined_df = pd.merge(df1, df2, how = how, left_on = [left], right_on = [right],)
    return joined_df


if __name__ == '__main__':
    #load the cleaned csv's from us census
    # all demographic data
    demo_df = pd.read_csv('../data/co-est2015-alldata-utf8-3142.csv')
    # all geographic data
    geo_df = pd.read_csv('../data/2015_Gaz_counties_national-utf8-3142.csv', dtype={'GEOID':str})

    # Loads the csv with the the bars with payroll
    dfb1 = processing('../data/BP_2015_00A1_with_ann_bar_utf8.csv', 'ESTAB', 'bar_big')
    # Loads the csv with the the bars without payroll
    dfb2 = processing('../data/NES_2015_00A2_with_ann_bar_utf8.csv', 'NESTAB', 'bar_small', all_numeric=False)

    # Loads the csv with the the hotels with payroll
    dfh1 = processing('../data/BP_2015_00A1_with_ann_hot_utf8.csv', 'ESTAB', 'hot_big')
    # Loads the csv with the the hotels without payroll
    dfh2 = processing('../data/NES_2015_00A2_with_ann_hot_utf8.csv', 'NESTAB', 'hot_small', all_numeric=False)

    # Loads the csv with the liquor stores
    dfls = processing('../data/BP_2015_00A1_with_ann_utf8.csv', 'ESTAB', 'liquor_stores')
    # Loads the csv with household income
    # dfls_clean = dfls['ESTAB']
    dfin = processing('../data/ACS_15_5YR_S1901_with_ann_utf8.csv', 'HC01_EST_VC13', 'household_median_income', all_numeric=False)
    # dfin.apply(pd.to_numeric)
    # print(dfin.dtypes)
    # # dfin['household_median_income'] = dfin['household_median_income'].astype(int)
    # Create new demo_df with only 2015 columns (the col used were selected by exploring the csv file in a notebook)
    demo_df2 = selec_col_df(demo_df, ['STNAME', 'CTYNAME', 'POPESTIMATE2015', 'NPOPCHG_2015'])

    # Create new geo_df2 with some of the columns of geo_df1 (the col used were selected by exploring the csv file in a notebook)
    geo_df2 = selec_col_df(geo_df, ['USPS', 'GEOID', 'NAME', 'ALAND_SQMI', 'INTPTLAT', 'INTPTLONG'])

    # Combine the two dataframes side by side
    # demo_geo_df = combine_dfs(geo_df2, demo_df2, 'NAME', 'CTYNAME')
    demo_geo_df = pd.concat([demo_df2, geo_df2], axis=1)


    # Generate new features based on the previous ones
    demo_geo_df['log_area_sqmi'] = np.log10(demo_geo_df['ALAND_SQMI'])
    demo_geo_df['log_pop_est_2015'] = np.log10(demo_geo_df['POPESTIMATE2015'])
    demo_geo_df['pop_den_2015'] = demo_geo_df['POPESTIMATE2015']/demo_geo_df['ALAND_SQMI']
    demo_geo_df['log_pop_den_2015'] = np.log10(demo_geo_df['pop_den_2015'])
    # demo_geo_df['pop_ch_10000_2015'] = 10000*demo_geo_df['NPOPCHG_2015']/demo_geo_df['POPESTIMATE2015']
    demo_geo_df['county_name'] = demo_geo_df['CTYNAME'] + ', ' + demo_geo_df['USPS']

    # Sort the columns
    sequence = ['STNAME', 'USPS', 'GEOID', 'county_name', 'ALAND_SQMI', 'log_area_sqmi', 'POPESTIMATE2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015', 'NPOPCHG_2015', 'pop_ch_10000_2015', 'INTPTLAT', 'INTPTLONG']
    sorted_df = demo_geo_df.reindex(columns = sequence)

    # Rename the columns
    sorted_df.columns = ['state_name', 'state_code', 'geo_id', 'county_name', 'area_sqmi', 'log_area_sqmi', 'pop_est_2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015', 'tot_pop_ch_2015', 'pop_ch_10000_2015', 'lat', 'long']

    # A reduced df with the features we think we want to keep right now
    demo_geo_df_2 = sorted_df[['geo_id', 'state_name', 'state_code', 'county_name', 'area_sqmi', 'log_area_sqmi', 'pop_est_2015', 'log_pop_est_2015', 'pop_den_2015', 'log_pop_den_2015']]

    # Merges all the hotels, income, liquor stores and bars to the main df
    hotels_df = pd.merge(dfh1, dfh2, on='geo_id', how='outer')
    bars_df = pd.merge(dfb1, dfb2, on='geo_id', how='outer')
    hot_bar_df = pd.merge(hotels_df, bars_df, on='geo_id', how='outer')
    inc_liqu = pd.merge(dfls, dfin, on='geo_id', how='right')
    hot_bar_liq_inc_df = pd.merge(inc_liqu, hot_bar_df, on='geo_id', how='left')
    all_merged_df = pd.merge(demo_geo_df_2, hot_bar_liq_inc_df, on='geo_id', how='outer')
    print(len(demo_geo_df_2),len(hotels_df),len(bars_df),len(hot_bar_df),len(inc_liqu),len(hot_bar_liq_inc_df),len(all_merged_df))

    # Makes the columns floats to be able to operate on them
    all_merged_df['hot_big'] = all_merged_df['hot_big'].astype(float)
    all_merged_df['bar_big'] = all_merged_df['bar_big'].astype(float)

    # Fills the NaN in with the min of the rows with values
    all_merged_df['bar_big'].fillna(all_merged_df['bar_big'].min(), inplace=True)
    all_merged_df['hot_big'].fillna(all_merged_df['hot_big'].min(), inplace=True)
    all_merged_df['liquor_stores'].fillna(all_merged_df['liquor_stores'].min(), inplace=True)

    # Fills the NaN in with the min of the rows with values
    all_merged_df['bar_small'].fillna(3, inplace=True)
    all_merged_df['hot_small'].fillna(3, inplace=True)

    # Replaces the D's and S's with 1 or 2 randomly with values
    all_merged_df.replace('D', np.random.randint(1,3), inplace=True)
    all_merged_df.replace('S', np.random.randint(1,3), inplace=True)

    # Creats columns with the total number of hotels and bars
    all_merged_df['hotels'] = all_merged_df['hot_big'] + all_merged_df['hot_small'].astype(float)
    all_merged_df['bars'] = all_merged_df['bar_big'] + all_merged_df['bar_small'].astype(float)

    all_merged_df['log_bars'] = np.log10(all_merged_df['bars'])
    all_merged_df['log_hotels'] = np.log10(all_merged_df['hotels'])

    all_merged_df_lin = all_merged_df[['geo_id', 'state_name', 'state_code', 'county_name', 'area_sqmi', 'pop_est_2015', 'pop_den_2015', 'hotels', 'liquor_stores', 'household_median_income', 'bars']]

    all_merged_df_log = all_merged_df[['geo_id', 'state_name', 'state_code', 'county_name', 'log_area_sqmi', 'log_pop_est_2015', 'pop_den_2015', 'log_hotels', 'household_median_income', 'bars']]

    # Saves the dataframe as a csv with all cols as a checkpoint
    sorted_df.to_csv('../data/2015_demo_geo.csv')

    hot_bar_df.to_csv('../data/2015_hot_bar_nofill.csv')

    all_merged_df.to_csv('../data/2015_master_sd_rnd_nan_to_min.csv')
    print(all_merged_df.columns)
    all_merged_df_lin.to_csv('../data/2015_lin_sd_rnd_nan_to_min.csv')
    print(all_merged_df_lin.columns)
    all_merged_df_lin.to_csv('../data/2015_log_sd_rnd_nan_to_min.csv')
    print(all_merged_df_log.columns)
