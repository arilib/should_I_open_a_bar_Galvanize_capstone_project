import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

def split_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    county_info_df = df.copy()
    y = df.pop('bars')
    X = df.drop(['geo_id', 'state_name', 'state_code', 'county_name'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
    cols = X_train.columns
    return(X_train, X_test, y_train, y_test, county_info_df, cols)


def lin_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression(fit_intercept=True)
    model = lr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    return(model, train_score, test_score)

def rid_reg(X_train, X_test, y_train, y_test):
    rr = Ridge(fit_intercept=True)
    model = rr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    return(model, train_score, test_score)

def geo_id_finder(county, state, df):
    if len(state) == 2:
        the_state = state.upper()
        state_df = df[df['state_code'] == the_state]
    else:
        the_state = state.lower().title()
        state_df = df[df['state_name'] == the_state]
    county_list = list(state_df['county_name'])
    for elem in county_list:
        m = re.search(county, elem, re.IGNORECASE)
        if m:
            req_county = elem
            break
    m = re.search('aleutians', req_county, re.IGNORECASE)
    if m and county != req_county:
        aleu = input('Are you interested in 1)Aleutians East Borough or 2)Aleutians West Census Area?\nEnter 1 or 2: ')
        if aleu == '1': req_county = 'Aleutians East Borough'
        elif aleu == '2': req_county = 'Aleutians West Census Area'

    index = state_df.index[state_df['county_name'] == req_county].tolist()[0]
    return(index, req_county)


def predict_bars(idx, county_info_df, model, cols):
        X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
        y = county_info_df.iloc[idx]['bars']
        y_pred = model.predict(X)[0]
        return(int(y_pred), int(y))



if __name__ == '__main__':
    filename = '../data/2015_toy_sd_1_5_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info_df, cols = split_data(filename)
    lr_model, lr_train_score, lr_test_score = lin_reg(X_train, X_test, y_train, y_test)
    print('Linear Regression Score\nTrain: {0}\nTest: {1}\n'. format(lr_train_score, lr_test_score))
    rr_model, rr_train_score, rr_test_score = rid_reg(X_train, X_test, y_train, y_test)
    print('Ridge Regression Score\nTrain: {0}\nTest: {1}\n'. format(rr_train_score, rr_test_score))

    state = input('Enter the state you are interested in:\n')
    county = input('\nEnter the county you are interested in:\n')
    index, req_county = geo_id_finder(county, state, county_info_df)
    pred_y, actual_y = predict_bars(index, county_info_df, rr_model, cols)
    print('\nAccording to this model, the number of bars {3} could suport is: {0}\n. The actual number of bars: {1}\nDifference: {2}\n'.format(pred_y, actual_y, pred_y - actual_y, req_county))
