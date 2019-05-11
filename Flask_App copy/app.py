import flask
from flask import request
from flask import render_template
import pickle
import pandas as pd
import numpy as np
import sklearn

app = flask.Flask(__name__, template_folder='templates')

# Loading the Models
with open('adamodel.pickle', 'rb') as f:
    adaModel = pickle.load(f)

with open('gbmodel.pickle', 'rb') as f:
    gbModel = pickle.load(f)

with open('rfmodel.pickle', 'rb') as f:
    rfModel = pickle.load(f)

# vector_dict = attributes = {
#     'AWND': 'Average daily wind speed',
#     'FMTM': 'Time of fastest mile',
#     'PRCP': 'Precipitation',
#     'SNOW': 'Snowfall',
#     'SNWD': 'Snow depth',
#     'TAVG': 'Average temperature',
#     'TMIN': 'Minimum temperature',
#     'TSUN': 'Total daily sunshine',
#     'WESD': 'Water equivalent of snow on the ground',
#     'WSFG': 'Peak guest wind speed',
#     'WV01': 'Fog, ice fog, or freezing fog in the vicinity',
#     'WT04': 'Ice pellets, sleet, snow pellets, or small hail',
#     'WT05': 'Hail (may include small hail)',
#     'WT06': 'Glaze or rime',
#     'WT09': 'Blowing or drifting snow',
#     'WT11': 'High or damaging winds',
#     'WT15': 'Freezing drizzle',
#     'WT17': 'Freezing rain',
#     'WT18': 'Snow, snow pellets, snow grains, or ice crystals',
#     'WT22': 'Ice fog or freezing fog'
# }
#
# vector_pos = [0, 0, 1, 10, 5, 25, 19, 0, 2, 20, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]
#
# vector_neg = [0, 0, 0, 0, 0, 60, 50, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# KNN.predict([vector_pos])
#
# KNN.predict([vector_neg])


@app.route('/')
# @app.route('/index')
def main():
    return render_template('index.html')


def ValuePredictor(to_predict_list, model):
    to_predict = [to_predict_list]
    result = model.predict(to_predict)
    return result


ValuePredictor([8.2, 83.0], adaModel)

# ValuePredictor(vector_neg)


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list, adaModel)
        prediction = round(100 * result[0], 3)
        return render_template("result.html", prediction=prediction)


# Run web app server
if __name__ == '__main__':
    app.run()
    # app.run(debug=True)
