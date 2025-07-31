import pandas as pd
import numpy as np

class MediaMixDataCleaner:
    def __init__(self, df: pd.DataFrame, upper_pct: float = 0.99):
        self.df = df.copy()
        self.upper_pct = upper_pct

    def impute_time_series(self, cols: list[str] | None = None):

        cols = cols or self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.df[cols] = self.df[cols].ffill().bfill()
        return self

    def winsorize_upper(self, cols: list[str]):

        for col in cols:
            if col not in self.df:
                continue
            cap = self.df[col].quantile(self.upper_pct)
            self.df[col] = np.minimum(self.df[col], cap)
        return self

    def clean(self, raw_kpi_cols: list[str], control_cols: list[str]):

        self.impute_time_series(raw_kpi_cols + control_cols)
        self.winsorize_upper(raw_kpi_cols)
        return self.df

