import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPrep:
    def __init__(self, df: pd.DataFrame, lam: float = 0.5):
        self.df = df.copy()
        self.lam = lam
        self.spend_cols = [c for c in self.df.columns if c.endswith("_spend")]
        self.baseline_cols = ["seasonality_index", "economic_index", "avg_discount_rate", "temperature"]

    def _adstock(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = x[i] + self.lam * out[i - 1]
        return out

    def transform(self) -> pd.DataFrame:
        drop_cols = [c for c in self.df.columns if c.endswith(("_impr", "_clicks", "_trps", "_spots", "_faces", "_reach"))]
        df = self.df.drop(columns=drop_cols)
        df[self.spend_cols] = np.log1p(df[self.spend_cols])
        for c in self.spend_cols:
            ad = self._adstock(df[c].values)
            df[f"{c}_ad"] = ad
            df[f"{c}_lag1"] = np.roll(ad, 1)
            df.loc[0, f"{c}_lag1"] = np.nan
        scaler = StandardScaler()
        df[self.baseline_cols] = scaler.fit_transform(df[self.baseline_cols])
        df = df.dropna().reset_index(drop=True)
        return df

