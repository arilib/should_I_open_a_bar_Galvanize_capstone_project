import os
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

states = ['California', 'Colorado', 'Florida', 'District of Columbia']

counties = ['Boulder', 'Denver', 'Jefferson']

def retr_model_info():
    with open('model_and_cols.pkl', 'rb') as f:
        county_info_df = pickle.load(f)
        model = pickle.load(f)
        cols = pickle.load(f)
    return(county_info_df, model, cols)



@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('form/index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('form/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['article_body'])
    pred = str(model.predict([data])[0])
    return render_template('form/predict.html', article=data, predicted=pred)

if __name__ == '__main__':
    county_info_df, model, cols = retr_model_info()
    app.run(host='0.0.0.0', debug=True)
