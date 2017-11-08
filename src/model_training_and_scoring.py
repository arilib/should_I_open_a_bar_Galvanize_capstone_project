import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge
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
    # print(county_info_df)
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

def model_running(model_code, X_train, X_test, y_train, y_test):
    if model_code == 'lr':
        lr = LinearRegression(fit_intercept=True)
        model = lr.fit(X_train,y_train)
    elif model_code == 'rr':
        rr = Ridge(fit_intercept=True)
        model = rr.fit(X_train,y_train)
    elif model_code == 'tr':
        tr = DecisionTreeRegressor(random_state = 7, max_depth=5)
        model = tr.fit(X_train, y_train)
    else: #model_code == fr:
        fr = RandomForestRegressor(n_estimators=20, max_depth=6, oob_score=True, n_jobs = -1, random_state = 7)
        model = fr.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_score = explained_variance_score(y_train.reshape(-1,1), y_train_pred.reshape(-1,1))
    test_score = explained_variance_score(y_test.reshape(-1,1), y_test_pred.reshape(-1,1))
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
    ax.set_xlim(right=400)
    ax.set_ylim(top=400)
    ax.set_xlabel('Actual number of bars')
    ax.set_ylabel('Predicted number of bars')
    ax.set_label(model_name)
    plt.title(model_name)
    plt.savefig('../figures/'+mp+'400'+filename[9:-3]+'png')
    # plt.show()
    return()

if __name__ == '__main__':
    warnings.simplefilter(action='ignore')
    filename = '../data/2015_lin_sd_rnd_nan_to_min.csv'
    X_train, X_test, y_train, y_test, county_info_df, cols = split_data(filename)

    list_of_models = [['lr', 'Linear Regression'], ['rr', 'Ridge Regression'], ['tr', 'Decision Tree Regressor'], ['fr', 'Random Forest Regressor']]

    score = 0
    for index in range(len(list_of_models)):
        model_code = list_of_models[index][0]
        model, train_score, test_score = model_running(model_code, X_train, X_test, y_train, y_test)
        if index > 0 and test_score > score:
            the_model = model
            the_score = test_score
        else:
            the_model = model
            the_score = test_score
        model_name = list_of_models[index][1]
        list_of_models[index].extend([model, train_score, test_score])
        print(model_name+' Score\nTrain: {0}\nTest: {1}\n'. format(round(train_score.mean(),3), round(test_score.mean(),3)))
        actual_pred_plot(model_name, model_code, model, X_test, y_test, filename)



    store_model_info('model_and_cols.pkl', county_info_df, the_model, cols)
    store_model_info('../web_app/model_and_cols.pkl', county_info_df, the_model, cols)
