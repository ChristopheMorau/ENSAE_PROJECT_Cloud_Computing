import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor



class Model:
    """Model for our predictor. Trains a XGBoost on a subset
    of relevant variables"""

    def __init__(self, path, labels_path) -> None:
        self.df = pd.read_csv(path)
        self.labels_path = labels_path
        self.reg = GradientBoostingRegressor()

    def variable_selection(self, save=False):
        """Selection of variable based on correlation to limit colinearity"""
        c = self.df[self.df.is_train].corr()

        arr = np.array(c)
        arr = arr-np.identity(c.shape[0])

        cor = pd.DataFrame(arr, columns=c.columns)

        dic = {}
        for idx,col in enumerate(cor.columns):
            l = cor[cor[col]>0.7]
            if len(l)>0:
                dic[idx] = list(l.index)
                
        to_keep=[]
        to_delete=[]
        for keys in dic.keys():
            if not(keys in to_delete):
                to_keep.append(keys)
                for item in dic[keys]:
                    if not(item in to_keep):
                        to_delete.append(item)

        to_delete = list(map(lambda x: list(cor.columns)[x], to_delete))
        okcolumns = list(set(self.df.columns)-set(to_delete))

        self.df = self.df[okcolumns]
        self.df = self.df.dropna()

        X = self.df[self.df.is_train].drop(columns='is_train')
        X_ = self.df[~self.df.is_train].drop(columns='is_train')
        y = pd.read_csv(self.labels_path).merge(
            X, on='level_0', how='inner')[['level_0', 'energy_consumption_per_annum']]
        X.to_csv('X.csv')
        X_.to_csv("X_.csv")
        y.to_csv('y.csv')
        return X, X_, y

    def fit(self, X, y):
        """fits the model with a validation set"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.reg.fit(X_train, y_train.energy_consumption_per_annum)
        print('training R²:', self.reg.score(X_train,y_train.energy_consumption_per_annum))

        print('validation set R²:', self.reg.score(X_test,y_test.energy_consumption_per_annum))

    def predict(self, X_):
        pred = self.reg.predict(X_)
        return pred

    def create_submission_file(self, submission):
        indexdf = X_[["level_0"]]
        indexdf = indexdf.reset_index().drop(columns=["index"])
        submission_df = pd.concat([
            indexdf,
            pd.DataFrame(submission, columns=["energy_consumption_per_annum"])
        ], axis=1)

        submission_df = submission_df.set_index("level_0")
        return submission_df
        