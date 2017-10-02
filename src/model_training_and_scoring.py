import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def split_data(filename):
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    county_info_df = df.copy()
    df = df[df['bars'] >= 3]
    df = df[df['hotels'] >= 3]
    y = np.log(df.pop('bars'))
    X = df.drop(['geo_id', 'state_name', 'state_code', 'county_name'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 7)
    cols = X_train.columns
    # print(cols)
    return(X_train, X_test, y_train, y_test, county_info_df, cols)

def lin_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression(fit_intercept=True)
    model = lr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train.values)
    y_test_pred = model.predict(X_test)
    train_score = cross_val_score(model, y_train.reshape(-1,1), y_train_pred.reshape(-1,1), cv = 5)
    test_score = cross_val_score(model, y_test.reshape(-1,1), y_test_pred.reshape(-1,1), cv = 5)
    return(model, train_score, test_score)

def rid_reg(X_train, X_test, y_train, y_test):
    rr = Ridge(fit_intercept=True)
    model = rr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = cross_val_score(model, y_train.reshape(-1,1), y_train_pred.reshape(-1,1), cv = 5)
    test_score = cross_val_score(model, y_test.reshape(-1,1), y_test_pred.reshape(-1,1), cv = 5)
    return(model, train_score, test_score)

def tr_reg(X_train, X_test, y_train, y_test):
    regr = DecisionTreeRegressor(random_state = 7)
    model = regr.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = cross_val_score(model, y_train.reshape(-1,1), y_train_pred.reshape(-1,1), cv = 5)
    test_score = cross_val_score(model, y_test.reshape(-1,1), y_test_pred.reshape(-1,1), cv = 5)
    return(model, train_score, test_score)

def rf_reg(X_train, X_test, y_train, y_test):
    regr = RandomForestRegressor(n_estimators=10, max_depth=2, oob_score=True, n_jobs = -1, random_state = 7)
    model = regr.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = cross_val_score(model, y_train.reshape(-1,1), y_train_pred.reshape(-1,1), cv = 5)
    test_score = cross_val_score(model, y_test.reshape(-1,1), y_test_pred.reshape(-1,1), cv = 5)
    return(model, train_score, test_score)

def store_model_info(county_info_df, model, cols):
    output = open('model_and_cols.pkl', 'wb')
    pickle.dump(county_info_df, output)
    pickle.dump(lr_model, output, -1)
    pickle.dump(cols, output, -1)
    output.close()
    return

def actual_pred_plot(mp, model, X_test, y_test, filename):
    y = y_test.reshape(-1,1)
    predicted = cross_val_predict(model, X_test, y, cv = 5)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0), alpha=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual number of bars')
    ax.set_ylabel('Predicted number of bars')
    ax.set_label(filename)
    plt.savefig('../figures/'+mp+filename[9:-3]+'png')
    return()

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    filename = '../data/2015_lin_sd_rnd_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info_df, cols = split_data(filename)

    list_of_models = [('lr', 'Linear Regression', 'lr_model'), ('rr', 'Ridge Regression', 'rr_model'), ('tr', 'Decision Tree Regressor', 'tr_model'), ('rr', 'Random Forest Regressor', 'rf_model')]

    for model in list_of_models:
        print(model[0], model[1], model[2])


    lr_model, lr_train_score, lr_test_score = lin_reg(X_train, X_test, y_train, y_test)
    rr_model, rr_train_score, rr_test_score = rid_reg(X_train, X_test, y_train, y_test)
    tr_model, tr_train_score, tr_test_score = tr_reg(X_train, X_test, y_train, y_test)
    rf_model, rf_train_score, rf_test_score = rf_reg(X_train, X_test, y_train, y_test)


    print(filename)
    print('Linear Regression Score\nTrain: {0}\nTest: {1}\n'. format(round(lr_train_score.mean(),3), round(lr_test_score.mean(),3)))
    actual_pred_plot('lr', lr_model, X_test, y_test, filename)

    print('Ridge Regression Score\nTrain: {0}\nTest: {1}\n'. format(round(rr_train_score.mean(),3), round(rr_test_score.mean(),3)))
    actual_pred_plot('rr', rr_model, X_test, y_test, filename)

    print('Decision Tree Regressor Score\nTrain: {0}\nTest: {1}\n'. format(round(rf_train_score.mean(),3), round(rf_test_score.mean(),3)))
    actual_pred_plot('tr', tr_model, X_test, y_test, filename)

    print('Random Forest Regressor Score\nTrain: {0}\nTest: {1}\n'. format(round(rf_train_score.mean(),3), round(rf_test_score.mean(),3)))
    actual_pred_plot('rf', rf_model, X_test, y_test, filename)

    store_model_info(county_info_df, rr_model, cols)
