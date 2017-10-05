import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score

def split_data(filename):
    # scaler=StandardScaler()
    df = pd.read_csv(filename)
    df = df.drop('Unnamed: 0', axis=1)
    county_info_df = df.copy()
    # df = df[df['bars'] >= 3]
    # df = df[df['hotels'] >= 3]
    y = np.array(df.pop('bars')).reshape(-1,1)
    X_features = df.drop(['geo_id', 'state_name', 'state_code', 'county_name'], axis=1)
    cols = X_features.columns
    X = np.array(X_features)
    # X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)
    print(cols)
    return(X_train, X_test, y_train, y_test, county_info_df, cols)

def lin_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression(fit_intercept=True)
    model = lr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = explained_variance_score(y_train.reshape(-1,1), y_train_pred.reshape(-1,1))
    test_score = explained_variance_score(y_test.reshape(-1,1), y_test_pred.reshape(-1,1))
    return(model, train_score, test_score)

def rid_reg(X_train, X_test, y_train, y_test):
    rr = Ridge(fit_intercept=True)
    model = rr.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = explained_variance_score(y_train.reshape(-1,1), y_train_pred.reshape(-1,1))
    test_score = explained_variance_score(y_test.reshape(-1,1), y_test_pred.reshape(-1,1))
    return(model, train_score, test_score)

def tr_reg(X_train, X_test, y_train, y_test):
    regr = DecisionTreeRegressor(random_state = 7, max_depth=3)
    model = regr.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = explained_variance_score(y_train.reshape(-1,1), y_train_pred.reshape(-1,1))
    test_score = explained_variance_score(y_test.reshape(-1,1), y_test_pred.reshape(-1,1))
    return(model, train_score, test_score)

def rf_reg(X_train, X_test, y_train, y_test):
    regr = RandomForestRegressor(n_estimators=20, max_depth=3, oob_score=True, n_jobs = -1, random_state = 7)
    model = regr.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = explained_variance_score(y_train.reshape(-1,1), y_train_pred.reshape(-1,1))
    test_score = explained_variance_score(y_test.reshape(-1,1), y_test_pred.reshape(-1,1))
    importances = regr.feature_importances_
    print(importances)
    return(model, train_score, test_score)

def store_model_info(filename, county_info_df, model, cols):
    output = open(filename, 'wb')
    pickle.dump(county_info_df, output)
    pickle.dump(model, output, -1)
    pickle.dump(cols, output, -1)
    output.close()
    return

def actual_pred_plot(model_name, mp, model, X_test, y_test, filename):
    y = y_test.reshape(-1,1)
    predicted = cross_val_predict(model, X_test, y)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0), alpha=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    # ax.set_xlim(right=400)
    # ax.set_ylim(top=400)
    ax.set_xlabel('Actual number of bars')
    ax.set_ylabel('Predicted number of bars')
    ax.set_label(model_name)
    plt.loglog
    plt.savefig('../figures/'+mp+filename[9:-3]+'png')
    return()

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    filename = '../data/2015_lin_sd_rnd_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info_df, cols = split_data(filename)

    # print(filename)

    list_of_models = [('lr', 'Linear Regression', 'lr_model'), ('rr', 'Ridge Regression', 'rr_model'), ('tr', 'Decision Tree Regressor', 'tr_model'), ('rr', 'Random Forest Regressor', 'rf_model')]

    # for model in list_of_models:
    #     print(model[0], model[1], model[2])


    lr_model, lr_train_score, lr_test_score = lin_reg(X_train, X_test, y_train, y_test)
    rr_model, rr_train_score, rr_test_score = rid_reg(X_train, X_test, y_train, y_test)
    tr_model, tr_train_score, tr_test_score = tr_reg(X_train, X_test, y_train, y_test)
    rf_model, rf_train_score, rf_test_score = rf_reg(X_train, X_test, y_train, y_test)

    model_name = 'Linear Regression'
    print(model_name+' Score\nTrain: {0}\nTest: {1}\n'. format(round(lr_train_score.mean(),3), round(lr_test_score.mean(),3)))
    actual_pred_plot(model_name, 'lr', lr_model, X_test, y_test, filename)

    model_name = 'Ridge Regression'
    print(model_name+' Score\nTrain: {0}\nTest: {1}\n'. format(round(rr_train_score.mean(),3), round(rr_test_score.mean(),3)))
    actual_pred_plot(model_name, 'rr', rr_model, X_test, y_test, filename)

    model_name = 'Decision Tree Regressor'
    print(model_name+' Score\nTrain: {0}\nTest: {1}\n'. format(round(tr_train_score.mean(),3), round(tr_test_score.mean(),3)))
    actual_pred_plot(model_name, 'tr', tr_model, X_test, y_test, filename)

    model_name = 'Random Forest Regressor'
    print(model_name+' Score\nTrain: {0}\nTest: {1}\n'. format(round(rf_train_score.mean(),3), round(rf_test_score.mean(),3)))
    actual_pred_plot(model_name, 'rf', rf_model, X_test, y_test, filename)

    store_model_info('model_and_cols.pkl', county_info_df, rf_model, cols)
    store_model_info('../web_app/model_and_cols.pkl', county_info_df, rf_model, cols)
    store_model_info('../web_app_0/model_and_cols.pkl', county_info_df, rf_model, cols)
