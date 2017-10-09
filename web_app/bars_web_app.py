from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm, Form
import pandas as pd
import numpy as np
import re
import pickle
import math

app = Flask(__name__)

def retr_model_info():
    with open('model_and_cols.pkl', 'rb') as f:
        county_info_df = pickle.load(f)
        model = pickle.load(f)
        cols = pickle.load(f)
    return(county_info_df, model, cols)

def index_finder(state, county_idx, df):
    state_df = df[df['state_name'] == state]
    county_list = list(state_df['county_name'])
    req_county = county_list[int(county_idx)]
    index = df.index[df['county_name'] == req_county]
    return(index, req_county)

# the calculation of err_down and err_up are modified code obtained from http://blog.datadive.net/prediction-intervals-for-random-forests/
def predict_bars(idx, county_info_df, model, cols, percentile=95):
    X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
    y = county_info_df.iloc[idx]['bars']
    y_pred = model.predict(X)
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(X)[0])
    err_down = (np.percentile(preds, (100 - percentile) / 2. ))
    err_up = (np.percentile(preds, 100 - (100 - percentile) / 2.))
    return(max(0,int(y_pred)), max(0,math.ceil(err_down)), math.floor(err_up), int(y))

def mark_status(pred, actual):
    if actual == 0:
        market_status, evalu = 'Very Unsatisfied', ' Opening a bar there may be a great idea.'
        return(market_status, evalu)
    r = pred/actual
    if r >= 1.5:
        market_status, evalu = 'very unsatisfied', ' Opening a bar there may be a great idea!'
        return(market_status, evalu)
    elif r >= 1.2:
        market_status, evalu = 'unsatisfied', ' Opening a bar there may be a good idea.'
        return(market_status, evalu)
    elif r >= 0.8:
        market_status, evalu = 'balanced', ''
        return(market_status, evalu)
    elif r >= 0.5:
        market_status, evalu = 'saturated', ' Opening a bar there may be a bad idea.'
        return(market_status, evalu)
    return('very saturated', ' Opening a bar there may be a very bad idea.')
@app.route('/', methods=['GET', 'POST'])
def home():
    '''Landing page with and explanation of the site'''
    return render_template('home.html')

@app.route('/advice/', methods=['GET','POST'])
def advice():
    state = request.form['state']
    if state == '':
        return render_template('pick_county.html', states=state_list)
    county_idx = request.form['county']
    if county_idx == '':
        return render_template('advice.html')
    idx, req_county = index_finder(state.split("'")[1], county_idx, county_info_df)
    pred, low_numb, high_numb, actual = predict_bars(idx, county_info_df, model, cols)
    market_status, evalu = mark_status(pred, actual)
    print(market_status)
    print(evalu)
    return render_template('advice.html', **locals())

@app.route('/pick_county/', methods=['GET','POST'])
def pick_county():
    return render_template('pick_county.html', states=state_list)


@app.route('/get_counties')
def get_counties():
    state = request.args.get('state')
    if state:
        state_county = county_info_df[county_info_df['state_name'] == state.split("'")[1]]
        data = [(idx, state_county.iloc[idx]['county_name'][:-4]) for idx in range(len(state_county))]
    return jsonify(data)


@app.route('/rankings/', methods=['GET', 'POST'])
def rankings():
    '''Page showing the rankings of best and worst counties and a map'''
    return render_template('rankings.html')

# @app.route('/map/', methods=['GET', 'POST'])
# def map():
#     '''Page showing a county level map'''
#     return render_template('map.html')


if __name__ == '__main__':
    county_info_df, model, cols = retr_model_info()
    states = county_info_df.state_name.unique()
    state_list = [(idx, states[idx]) for idx in range(len(states))]
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
