import os
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm, Form
from flask_wtf.csrf import CSRFProtect
from wtforms import SelectField, HiddenField, SubmitField
from wtforms.validators import DataRequired
from collections import Counter
import pandas as pd
import numpy as np
import re
import pickle
from collections import defaultdict

app = Flask(__name__)
csrf = CSRFProtect()
class PickCounty(FlaskForm):
    form_name = HiddenField('Form Name')
    state = SelectField('State:', validators=[DataRequired()], id='select_state')
    county = SelectField('County:', validators=[DataRequired()], id='select_county')
    submit = SubmitField('Select County!')

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

def make_dict():
    county_dict = defaultdict(list)
    for k,v in county_info_df[['state_name','county_name']].values:
        county_dict[k].append(v.split(',')[0])
    return county_dict

def predict_bars(idx, county_info_df, model, cols):
        X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
        y = county_info_df.iloc[idx]['bars']
        y_pred = model.predict(X)[0]
        return(int(y_pred), int(y))

# # the calculation of err_down and err_up are modified code obtained from http://blog.datadive.net/prediction-intervals-for-random-forests/
# def predict_bars(idx, county_info_df, model, cols, percentile=95):
#     X = np.asarray(county_info_df.iloc[idx][cols]).reshape(1,-1)
#     y = county_info_df.iloc[idx]['bars']
#     y_pred = model.predict(X)
#     preds = []
#     for pred in model.estimators_:
#         preds.append(pred.predict(X)[0])
#     err_down = (np.percentile(preds, (100 - percentile) / 2. ))
#     err_up = (np.percentile(preds, 100 - (100 - percentile) / 2.))
#     return(int(y_pred), math.ceil(err_down), math.floor(err_up), int(y))


@app.route('/', methods=['GET', 'POST'])
def home():
    '''Landing page with and explanation of the site'''
    return render_template('home.html')

# @app.route('/pick_county/', methods=['GET', 'POST'])
# def pick_county():
#     '''Page showing the rankings of best and worst counties and a map'''
#     return render_template('pick_county.html')


# @app.route('/create/', methods=['GET','POST'])
# def create():
#     mySQL2 = SelectCustomer(session['ID'])
#     global sessioncur
#     try:
#         form = CreateinvoiceForm(request.form)
#         if request.method == 'POST' and form.validate():
#             customer = request.form.get('customer')
#             goodsrec = request.form.get('goodsrec')
#             # do stuff with submitted form...
#     return render_template("pick_county.html", form=form,  mySQL2 = mySQL2)
#
#
# @app.route('/get_goods_receivers/')
# def get_goods_receivers():
#     customer = request.args.get('customer')
#     print(customer)
#     if customer:
#         # c = connection()
#         # customerID = c.execute("SELECT Cm_Id FROM customer WHERE Cm_name = %s LIMIT 1", [customer])
#         # customerID = c.fetchone()[0]
#         print customerID
#         c.execute("SELECT * FROM goodsrec WHERE Gr_Cm_id = %s", [customerID])
#         mySQL8 = c.fetchall()
#         c.close()
#         # x[0] here is Gr_id (for application use)
#         # x[3] here is the Gr_name field (for user display)
#         data = [{"id": x[0], "name": x[3]} for x in mySQL8]
#         print(data)
#     return jsonify(data)
#

@app.route('/pick_county/', methods=['GET', 'POST'])
def pick_county():
    form = PickCounty(form_name='PickCounty')
    states = county_info_df.state_name.unique()
    form.state.choices = [(idx, states[idx]) for idx in range(len(states))]
    form.county.choices = [(idx, county_info_df.iloc[idx]['county_name'][:-4]) for idx in range(len(county_info_df))]
    if request.method == 'GET':
        return render_template('pick_county.html', form=form)
    if form.validate_on_submit() and request.form['form_name'] == 'PickCounty':
        print(state, county)
        flash('state: %s, county: %s' % (form.state.data, form.county.data))
    return redirect(url_for('pick_county'))

@app.route('/_get_counties/')
def _get_counties():
    state = request.args.get('state', type=str)
    if state:
        state_county = county_info_df[county_info_df['state_name'] == state]
        counties = [(idx, state_county.iloc[idx]['county_name'][:-4]) for idx in range(len(state_county))]
    return jsonify(counties)


# @app.route('/pick_county/', methods=['GET', 'POST'])
# @csrf.exempt
# def pick_county():
#     form = PickCounty(form_name='PickCounty')
#     # form.state.choices = [(row.ID, row.Name) for row in State.query.all()]
#     # form.county.choices = [(row.ID, row.Name) for row in County.query.all()]
#     states = county_info_df.state_name.unique()
#     state_list = [{state: state} for state in states]
#     form.state.choices = [{state: state} for state in state_list]
#     form.county.choices = [{'size':'big'}, {'size':'medium'}, {'size':'small'}]
#
#     if request.method == 'GET':
#         return render_template('pick_county.html', form=form)
#     if form.validate_on_submit() and request.form['form_name'] == 'PickCounty':
#         # code to process form
#         flash('state: %s, county: %s' % (form.state.data, form.county.data))
#     # return redirect(url_for('pick_county'))
#     page = 'In {0}, there are {2} bars.<br><br>Our model indicates that there can be {1}'
#     return page.format(text2, pred, actual)
#
# @app.route('/_get_counties/')
# def _get_counties():
#     state = request.args.get('state', '01', type=str)
#     counties = [(row.ID, row.Name) for row in County.query.filter_by(state=state).all()]
#     return jsonify(counties)

@app.route('/rankings/', methods=['GET', 'POST'])
def rankings():
    '''Page showing the rankings of best and worst counties and a map'''
    return render_template('rankings.html')

@app.route('/map/', methods=['GET', 'POST'])
def map():
    '''Page showing a conty level map'''
    return render_template('map.html')

app.secret_key = os.environ['FLASK_SECRET_KEY']

if __name__ == '__main__':
    county_info_df, model, cols = retr_model_info()
    county_dict = make_dict()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
