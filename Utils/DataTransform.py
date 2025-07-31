import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

class MMMTransformer:
    def __init__(self, df, spend_cols, log_cols, adstock_map, lag_map, decay=0.5):
        self.df = df.copy()
        self.spend_cols = spend_cols
        self.log_cols = log_cols
        self.adstock_map = adstock_map  
        self.lag_map = lag_map          
        self.decay = decay

    def standardize(self, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = []
        cols = self.df.select_dtypes(include=np.number).columns.difference(exclude_cols)
        scaler = StandardScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])
        return self

    def log_transform(self):
        for c in self.log_cols:
            self.df[c] = np.log1p(self.df[c])
        return self

    def adstock(self):
        for c, lam in self.adstock_map.items():
            out = np.zeros(len(self.df))
            for i, x in enumerate(self.df[c]):
                out[i] = x if i == 0 else x + lam * out[i-1]
            self.df[f"{c}_ad"] = out
        return self

    def add_lags(self):
        for c, lags in self.lag_map.items():
            for l in lags:
                self.df[f"{c}_lag{l}"] = self.df[c].shift(l)
        return self

    def run(self):
        return self.log_transform().adstock().add_lags().standardize(exclude_cols=["weekly_revenue"]).df
