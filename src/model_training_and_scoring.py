import pandas as pd
import numpy as np
import pickle
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

def store_model_info(county_info_df, model, cols):
    output = open('model_and_cols.pkl', 'wb')
    pickle.dump(county_info_df, output)
    pickle.dump(lr_model, output, -1)
    pickle.dump(cols, output, -1)
    output.close()
    return

if __name__ == '__main__':
    filename = '../data/2015_toy_sd_rnd_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info_df, cols = split_data(filename)
    lr_model, lr_train_score, lr_test_score = lin_reg(X_train, X_test, y_train, y_test)



    print('Linear Regression Score\nTrain: {0}\nTest: {1}\n'. format(lr_train_score, lr_test_score))
    rr_model, rr_train_score, rr_test_score = rid_reg(X_train, X_test, y_train, y_test)
    print('Ridge Regression Score\nTrain: {0}\nTest: {1}\n'. format(rr_train_score, rr_test_score))

    store_model_info(county_info_df, rr_model, cols)
