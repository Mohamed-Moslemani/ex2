import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


class VIFReducer:
    def __init__(self, df: pd.DataFrame, target: str = "weekly_revenue"):
        self.y = df[target]
        X = df.drop(columns=[c for c in (target, "week") if c in df.columns])
        nunique = X.nunique()
        X = X.loc[:, nunique > 1]
        X = X.T.drop_duplicates().T

        self.X0 = X

    def reduce(
        self,
        thresh: float = 5.0,
        keep: list[str] | None = None,       
    ) -> pd.DataFrame:
        keep = set(keep or [])

        Xc = sm.add_constant(self.X0.select_dtypes("number"), has_constant="add")

        while Xc.shape[1] > 2:                
            scores = {
                col: (
                    np.inf if col in keep else
                    vif(Xc.values, i)
                )
                for i, col in enumerate(Xc.columns)
                if col != "const"
            }

            worst, worst_vif = max(scores.items(), key=lambda kv: kv[1])

            if worst_vif <= thresh:        
                break
            Xc = Xc.drop(columns=worst)      

        self.X_reduced = Xc.drop(columns="const")
        return self.X_reduced

    def vif_for(self, col: str) -> float:
        if col not in self.X0.columns:
            raise ValueError(f"Column '{col}' not found in predictors.")

        Xc = sm.add_constant(self.X0.select_dtypes("number"), has_constant="add")
        cols = Xc.columns.tolist()
        idx = cols.index(col)
        vif_value = vif(Xc.values, idx)
        print(f"VIF for '{col}': {vif_value:.2f}")
        return vif_value
