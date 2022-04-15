import numpy as np
import DiCE
import dice_ml
from dice_ml import Dice
import pandas as pd
import joblib
from IPython.display import display


class DiceExplainer:
    def __init__(self):
        pass

    def explain(model, index):
        # Import and Preprocessing of data (same as in model.py)
        df = pd.read_csv('../Datasets/House Price Prediction/01_House Sales in King County USA/kc_house_data.csv', index_col=0)
        #model = joblib.load('model.pkl')
        #alternative bins:
        #bins = pd.IntervalIndex.from_tuples(
        #    [(0, 300000), (300000, 350000), (350000, 400000), (400000, 450000), (450000, 500000), (500000, 550000),
        #     (550000, 600000), (600000, 650000), (650000, 700000), (700000, 1000000000)])

        bins = pd.IntervalIndex.from_tuples([(0, 300000), (300000, 400000), (400000, 500000), (500000, 700000), (700000, 1000000000)])
        df['priceframe'] = pd.cut(df['price'], bins=bins).cat.codes
        target = df['priceframe']
        data = df.iloc[:, 2:]


        #data = df.drop(['price', 'date','yr_built' , 'waterfront', 'view', 'condition', 'grade', 'sqft_basement', 'sqft_above', 'yr_renovated', 'sqft_lot15', 'sqft_living15', 'zipcode', 'lat', 'long'], axis=1)
        print(data.head())
        print()

        #Initialize explainer
        index = int(index)
        continuous_features = data.columns.tolist()

        d_data = dice_ml.Data(dataframe=data, continuous_features=['bathrooms', 'sqft_living'], outcome_name='priceframe')
        d_model = dice_ml.Model(model=model, backend='sklearn', model_type='classifier')
        d_explainer = dice_ml.Dice(d_data, d_model)

        single_output = data[index:index+1].drop('priceframe', axis=1)
        print(single_output)
        multiple_output = data[index:index+10].drop('priceframe', axis=1)
        counterf = d_explainer.generate_counterfactuals(multiple_output, total_CFs=7, desired_class=1) #, features_to_vary=['bedrooms', 'sqft_living', 'bathrooms'])
        counterf.visualize_as_dataframe()

        result = {}
        #expmap = exp.as_map()
        #for key in expmap.keys():
        #    objects = []
        #    for entry in expmap[key]:
        #        obj = {"id": int(entry[0]), "name": data.keys()[entry[0]], "value": entry[1]}
        #        objects.append(obj)
        #    result[str(key)] = {'values': objects, 'probability': model.predict_proba(data.iloc[index:index+1])[0].tolist()[key]}
        return result

