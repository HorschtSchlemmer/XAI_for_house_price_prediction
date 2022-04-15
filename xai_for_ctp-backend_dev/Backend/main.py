from flask import Flask, jsonify, request
import joblib
import json
from explainer import Explainer
from model import Model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/model/upload')
def uploadmodel():
    return "Model is uploaded"

@app.route('/model/train')
def train():
    datapath = '../Datasets/House Price Prediction/01_House Sales in King County USA/kc_house_data.csv'
    score = Model.train(datapath)
    return {"score": score}

@app.route('/explain/batch')
def exp_batch():
    model = joblib.load('model.pkl')
    index = request.args.get('index')
    explanations = []
    samples = []
    for id in range(int(index)):
        res = Explainer.explain(model, id)
        explanations.append(res['explanations'])
        samples.append(res['task']['data']['sample'])
    result = res
    result['explanations'] = explanations
    result['task']['data']['sample'] = samples
    return {"result": result}
    #return index#str(exp.as_list())

@app.route('/explain/single')
def exp_single():
    model = joblib.load('model.pkl')
    index = request.args.get('index')
    result = {"result": Explainer.explain(model, index)}
    return result
    #return index#str(exp.as_list())

@app.route('/explain/dice/single')
def exp_dice_single():
    model = joblib.load('model.pkl')
    index = request.args.get('index')
    result = {'result': DiceExplainer.explain(model, index)}
    return result


app.run(debug=True, use_reloader=False, host='0.0.0.0')
