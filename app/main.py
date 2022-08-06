# import requirements needed
import os
from flask import Flask, render_template, redirect, url_for, request, session
from utils import get_base_url
import pandas as pd
from xgboost import XGBRegressor
import pickle

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')
    
app.secret_key = os.urandom(64)

# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')

@app.route(f'{base_url}/analysis')
def analysis():
    return render_template('analysis.html')

@app.route(f'{base_url}/interactive')
def interactive():

    if 'prediction' in session:
        prediction = f"Our model predicts consumer inflation to be: {float(session['prediction']):.2f} percent"
    else:
        prediction = "The model prediction will be generated here!"
    
    return render_template('interactive-plots.html', prediction = prediction)

@app.route(f'{base_url}/predict', methods=["POST"])
def predict():

    # get user input forms data
    country = request.form['country_pred']

    inputs = pd.DataFrame({
        'Lending interest rate (%)': [float(request.form['lending_interest'])],
        'Unemployment, total (% of total labor force) (national estimate)': [float(request.form['unemployment'])],
        'GDP Growth': [float(request.form['gdp_growth'])]
    })

    # load model
    model = XGBRegressor()
    model.load_model(f"models/{country}.json")

    # make prediction
    prediction = model.predict(inputs)[0]

    # save prediction to session data
    session['prediction'] = str(prediction)

    return redirect(url_for('interactive'))

@app.route(f'{base_url}/generate_chart', methods=["POST"])
def generate_chart():

    country = request.form['country_chart']
    x = request.form['x']
    y = request.form['y']
    trendline = request.form['trendline']

    file_name = f"{country}-{x}-{y}-{trendline}.html"

    session['chart_file'] = file_name

    return redirect(url_for('interactive'))

@app.route(f'{base_url}/render_chart')
def render_chart():
    
    if 'chart_file' in session:
        file_name = session['chart_file']
    else:
        file_name = "Australia-Year-Year-ols.html"
    
    return render_template(file_name)

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc3.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
