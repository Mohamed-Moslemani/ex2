import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


class EDA:
    def __init__(self, df: pd.DataFrame, time_col: str = "week", target: str = "weekly_revenue"):
        self.df = df.copy()
        self.time_col= time_col
        self.target=target

        self.media_cols=[c for c in df.columns if c.endswith("_spend")]
        self.clicks_cols= [c for c in df.columns if c.endswith("_clicks")]
        self.impr_cols = [
            c
            for c in df.columns
            if any(token in c for token in ["_impr", "_trps", "_spots", "_reach", "_faces"])
        ]
        self.other_cols = [
            c
            for c in ["seasonality_index", "economic_index", "avg_discount_rate", "temperature"]
            if c in df.columns
        ]
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols=df.select_dtypes(exclude="number").columns.tolist()
    def missing_report(self):
        null_cnt = self.df.isna().sum()
        null_pct = null_cnt.mul(100/len(self.df))
        report = (
            pd.DataFrame({"missing": null_cnt, "%_missing": null_pct})
            .query("missing > 0")
            .sort_values("%_missing", ascending=False)
        )
        return report

    def plot_missing_heatmap(self):
        plt.figure(figsize=(12, 5))
        sns.heatmap(self.df.isna(), cbar=False, yticklabels=False)
        plt.title("Missing‑value pattern")
        plt.tight_layout()
        plt.show()

    def numeric_summary(self) -> pd.DataFrame:
        return self.df[self.num_cols].describe().T

    def outlier_summary(self,iqr_mult:float=1.5):
        stats = {}
        for col in self.num_cols:
            q1, q3 = np.percentile(self.df[col].dropna(), [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
            n_out = self.df[(self.df[col] < lo) | (self.df[col] > hi)][col].count()
            stats[col] = {"lower": lo, "upper": hi, "n_outliers": n_out}
        return pd.DataFrame(stats).T.sort_values("n_outliers", ascending=False)

    def plot_distributions(
            self,
            cols=None,
            bins=30,
            n_cols=4,
            save_path: str = "distributions.png",
    ):
        cols = cols or self.num_cols
        n = len(cols)
        n_rows = -(-n // n_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4, n_rows * 3),
            constrained_layout=True,
        )
        axes = axes.flatten()

        for ax_hist, col in zip(axes, cols):
            sns.histplot(
                self.df[col],
                bins=bins,
                kde=True,
                stat="density",
                line_kws={'color': 'black'},
                ax=ax_hist,
            )
            ax_hist.set_title(col, fontsize=9)

            ax_box = inset_axes(
                ax_hist,
                width="100%", height="20%",
                loc="lower center",
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=ax_hist.transAxes,
                borderpad=0,
            )
            sns.boxplot(
                x=self.df[col],
                ax=ax_box,
                orient="h",
                color="red",
                width=0.6,
            )
            ax_box.set_axis_off()

        for ax in axes[n:]:
            ax.set_visible(False)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


    def corr_heatmap(self, method: str = "pearson"):
        corr= self.df[self.num_cols].corr(method=method)
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, fmt=".2f", annot=False)
        plt.title(f"Correlation matrix – {method}")
        plt.tight_layout()
        plt.show()

    def corr_matrix(self, method: str = "pearson") -> pd.DataFrame:
        return self.df[self.num_cols].corr(method=method)

    def scatter_vs_target(self, cols: List[str], save_path: str | None = None) -> None:
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()
        for ax, col in zip(axes, cols):
            sns.scatterplot(x=self.df[col], y=self.df[self.target], ax=ax, s=25, alpha=0.7)
            ax.set_title(f"{col} vs {self.target}")
        for ax in axes[n:]:
            ax.remove()
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

 
    def plot_spend_trend(self, online: bool = True) -> None:
        online_keys = ["facebook", "instagram", "youtube"]
        cols = (
            [c for c in self.media_cols if any(k in c for k in online_keys)]
            if online
            else [c for c in self.media_cols if not any(k in c for k in online_keys)]
        )
        title = "Online" if online else "Offline"
        if not cols:
            print("No matching spend columns found.")
            return
        plt.figure(figsize=(12, 5))
        self.df.set_index(self.time_col)[cols].plot()
        plt.title(f"{title} media spend over time")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def total_spend_bar(self):
        s = self.df[self.media_cols].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 4))
        sns.barplot(x=s.index, y=s.values)
        plt.title("Total spend per media channel")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_monthly_average(self):
        df = self.df.copy()
        if not np.issubdtype(df[self.time_col].dtype, np.datetime64):
            df[self.time_col] = pd.to_datetime(df[self.time_col])
        monthly = (
            df.set_index(self.time_col)[self.target]
            .resample("M")
            .mean()
            .rename("monthly_avg")
        )
        plt.figure(figsize=(11, 4))
        plt.plot(monthly.index, monthly.values, marker="o")
        plt.title(f"{self.target} – monthly average")
        plt.xticks(monthly.index, monthly.index.strftime("%b‑%y"), rotation=45)
        plt.tight_layout()
        plt.show()
