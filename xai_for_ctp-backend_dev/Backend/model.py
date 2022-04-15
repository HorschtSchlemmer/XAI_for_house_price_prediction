import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class Model:
    def train(datapath):
        #Import Dataset
        df = pd.read_csv(datapath, index_col = 0)

        #classify prices into clusters
        bins = pd.IntervalIndex.from_tuples([(0, 300000), (300000, 400000), (400000, 500000), (500000, 700000), (700000, 1000000000)])
        df['priceframe'] = pd.cut(df['price'], bins=bins).cat.codes

        #Split target and data
        target = df['priceframe']
        data = df.iloc[:,2:-1]

        #Split train end test set
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train)

        #Make predictions
        y_pred = clf.predict(X_test)

        print(accuracy_score(y_test, y_pred))

        #export model
        joblib.dump(clf, './model.pkl')
        return accuracy_score(y_test, y_pred)