import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
import lime
from lime import lime_tabular
import joblib

class Explainer:
    def __init__(self):
        pass

    def explain(model, index):
        # Import and Preprocessing of data (same as in model.py)
        df = pd.read_csv('../Datasets/House Price Prediction/01_House Sales in King County USA/kc_house_data.csv', index_col=0)
        bins = pd.IntervalIndex.from_tuples([(0, 300000), (300000, 400000), (400000, 500000), (500000, 700000), (700000, 1000000000)])
        df['priceframe'] = pd.cut(df['price'], bins=bins).cat.codes

        target = df['priceframe']
        data = df.iloc[:, 2:-1]

        #Initialize explainer
        index = int(index)
        explainer = lime_tabular.LimeTabularExplainer(data.to_numpy(), kernel_width=5)
        exp = explainer.explain_instance(data.iloc[index], model.predict_proba, num_features=10, top_labels=5)
        expmap = exp.as_map()
        result = {}
        feature_importance = []
        sample = {}
        samples = []
        target_values = [{'name': '< 300000)', 'value': 0},
                         {'name': '[300000, 400000)', 'value': 1},
                         {'name': '[400000, 500000)', 'value': 2},
                         {'name': '[500000, 700000)', 'value': 3},
                         {'name': '< 700000', 'value': 4}]
        for key in data.keys().tolist():
            sample[key] = data.iloc[index, :][key]
        samples.append(sample)
        for key in expmap.keys():
            objects = []
            for entry in expmap[key]:
                obj = {"id": int(entry[0]), "name": data.keys()[entry[0]], "value": entry[1]}
                objects.append(obj)
            feature_importance.append(
                {'class': str(key), 'probability': model.predict_proba(data.iloc[index:index + 1])[0].tolist()[key],
                 'featureImportance': objects})
        result = {
            'task': {'data': {'featurenames': data.keys().tolist(), 'targetFeature': 'priceframe', 'sample': samples},
                     'classes': target_values}, 'explanations': feature_importance,
            'config': {'dataType': 'table', 'taskType': 'classification'}}
        return result