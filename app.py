# Import modules needed.
import ml_part

from flask import Flask
from flask import request
from flask import render_template
from cat_features import *

cat_factors = [producers, models, fuels, transmissions, powertrains, bodies, regions, conditions, safeties,
               comforts, multimedias, others, tags]

# Constants
MAIN_PAGE = 'main.html'
PRICE_PAGE = 'price.html'

# Create a Flask (WSGI) application as an instance of Flask class.
app = Flask(__name__)  # __name__ says Flask, that we need to look for additional files (i.e. templates folder).


@app.route('/')
def home():
    return render_template(MAIN_PAGE, cat_factors=cat_factors)


@app.route('/get-price', methods=['POST'])
def get_price():
    data = dict()
    data['Producer'] = request.form.get('producer')
    data['Model'] = request.form.get('model')
    data['Year'] = request.form.get('year')
    data['Mileage'] = request.form.get('mileage')
    data['Volume'] = request.form.get('volume')
    data['Fuel'] = request.form.get('fuel')
    data['Transmission'] = request.form.get('transmission')
    data['Powertrain'] = request.form.get('powertrain')
    data['Description'] = request.form.get('description')
    data['Body'] = request.form.get('body')
    data['Region'] = request.form.get('region')
    data['Condition'] = request.form.getlist('condition')
    data['Safety'] = request.form.getlist('safety')
    data['Comfort'] = request.form.getlist('comfort')
    data['Multimedia'] = request.form.getlist('multimedia')
    data['Other'] = request.form.getlist('other')
    data['Tags list'] = request.form.getlist('tags')
    data['Doors'] = request.form.get('doors')
    data['Seats'] = request.form.get('seats')
    for key in data:
        if type(data[key]) == list:
            data[key] = ', '.join(data[key])
    print(data)

    pred_prices = ml_part.predict_price(data)
    for idx in range(len(pred_prices)):
        pred_prices[idx] = int(pred_prices[idx])

    return render_template(PRICE_PAGE, input=data, prices=pred_prices)


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False  # To show correctly cyrillic symbols.
    app.run(debug=False)
