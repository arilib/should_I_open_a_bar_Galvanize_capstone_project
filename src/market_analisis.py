import pandas as pd
import numpy as np
import re
import pickle
import math
from sklearn.ensemble import RandomForestRegressor

def retr_model_info():
    with open('model_and_cols.pkl', 'rb') as f:
        county_info_df = pickle.load(f)
        model = pickle.load(f)
        cols = pickle.load(f)
    return(county_info_df, model, cols)

def index_finder(county, state, df):
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

<<<<<<< HEAD
# the calculation of err_down and err_up are modified code obtained from http://blog.datadive.net/prediction-intervals-for-random-forests/
=======
# def predict_bars(idx, county_info_df, model, cols):
#     X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
#     y = county_info_df.iloc[idx]['bars']
#     y_pred = model.predict(X)[0]
#     return(int(y_pred), int(y))
>>>>>>> cv
def predict_bars(idx, county_info_df, model, cols, percentile=95):
    X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
    y = county_info_df.iloc[idx]['bars']
    y_pred = model.predict(X)
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(X)[0])
    err_down = (np.percentile(preds, (100 - percentile) / 2. ))
    err_up = (np.percentile(preds, 100 - (100 - percentile) / 2.))
    return(int(y_pred), math.ceil(err_down), math.floor(err_up), int(y))


if __name__ == '__main__':

    county_info_df, model, cols = retr_model_info()
    state = input('Enter the state you are interested in:\n')
    county = input('\nEnter the county you are interested in:\n')
    index, req_county = index_finder(county, state, county_info_df)
    pred_y, lower_y, upper_y, actual_y = predict_bars(index, county_info_df, model, cols)
    print('\nAccording to this model, the number of bars {3} could suport is between {4} and {5} bars\n The actual number of bars: {1}\nDifference: {2}\n'.format(min(0,pred_y), actual_y, pred_y - actual_y, req_county, min(0,lower_y), upper_y))
