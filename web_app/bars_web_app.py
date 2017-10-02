import os
from flask import Flask, render_template, request, jsonify
from wtforms import Form , SelectField
from collections import Counter
import pandas as pd
import numpy as np
import re
import pickle

app = Flask(__name__)

# class PickCounty(Form):
#     form_name = HiddenField('Form Name')
#     state = SelectField('State:', validators=[DataRequired()], id='select_state')
#     county = SelectField('County:', validators=[DataRequired()], id='select_county')
#     submit = SubmitField('Select County!')

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
    index = state_df.index[state_df['county_name'] == req_county].tolist()[0]
    return(index, req_county)

def predict_bars(idx, county_info_df, model, cols):
        X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
        y = county_info_df.iloc[idx]['bars']
        y_pred = model.predict(X)[0]
        return(int(y_pred), int(y))



@app.route('/', methods=['GET'])
def index():
    page = "Should I open a bar?"
    '''Landing page with and explanation of the page'''
    return render_template('jumbotron.html', title="Should I Open a Bar?")

@app.route('/pick_county/', methods=['GET', 'POST'])
def pick_county():
    form = PickCounty(form_name='PickCounty')
    # form.state.choices = [(row.ID, row.Name) for row in State.query.all()]
    # form.county.choices = [(row.ID, row.Name) for row in County.query.all()]
    states = county_info_df.state_name.unique()
    state_list = [{state: state} for state in states]
    form.state.choices = [{state: state} for state in state_list]
    form.county.choices = [{'size':'big'}, {'size':'medium'}, {'size':'small'}]

    if request.method == 'GET':
        return render_template('pick_county.html', form=form)
    if form.validate_on_submit() and request.form['form_name'] == 'PickCounty':
        # code to process form
        flash('state: %s, county: %s' % (form.state.data, form.county.data))
    # return redirect(url_for('pick_county'))
    page = 'In {0}, there are {2} bars.<br><br>Our model indicates that there can be {1}'
    return page.format(text2, pred, actual)

@app.route('/_get_counties/')
def _get_counties():
    state = request.args.get('state', '01', type=str)
    counties = [(row.ID, row.Name) for row in County.query.filter_by(state=state).all()]
    return jsonify(counties)

@app.route('/rank_and_map', methods=['GET', 'POST'])
def predict():
    '''Page showing the rankings of best and worst counties and a map'''
    data = str(request.form['article_body'])
    pred = str(model.predict([data])[0])
    return render_template('form/predict.html', article=data, predicted=pred)

if __name__ == '__main__':
    county_info_df, model, cols = retr_model_info()
    app.run(host='0.0.0.0', port=5000, debug=True)
