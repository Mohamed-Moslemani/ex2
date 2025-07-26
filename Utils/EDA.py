import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

####EDA CLASS THAT INCLUDES MLTIPLE EDA STEPS 
## multiple methods to explore and understand the data
class Eda:
  def __init__(self,data):
   self.df = data.copy()
   self.media_cols = [c for c in self.df.columns if '_spend' in c]
   self.clicks_cols = [c for c in self.df.columns if '_clicks' in c]
   self.impr_cols= [c for c in self.df.columns if '_impr' in c or '_trps' in c or '_spots' in c or '_reach' in c or '_faces' in c]
   self.other_cols = ['seasonality_index','economic_index','avg_discount_rate','temperature']
   
  def shape_nulls(self):
   print('data shape:',self.df.shape)
   print('\nmissing vals:\n',self.df.isnull().sum()[self.df.isnull().sum()>0])

  def statz(self):
   desc=self.df.describe()
   print('\nsummary stats:\n',desc)

  def histss(self):
   nums=self.df.select_dtypes(include='number').columns
   for c in nums:
    fig, ax = plt.subplots(figsize=(6,3))
    self.df[c].hist(ax=ax, bins=20)
    ax.set_title('hist of '+c)
    plt.tight_layout()
    plt.show()

  def time_seri(self):
   fig, ax= plt.subplots()
   ax.plot(self.df['week'],self.df['weekly_revenue'])
   ax.set_title('weekly revenue ts')
   ax.tick_params(axis='x',rotation=90)
   plt.tight_layout()
   plt.show()

  def spend_vs_rev(self):
   for c in self.media_cols:
     fig, ax=plt.subplots()
     sns.scatterplot(x=self.df[c],y=self.df['weekly_revenue'],ax=ax)
     ax.set_title(c+' vs revenue')
     plt.tight_layout()
     plt.show()

  def clicks_vs_rev(self):
    for c in self.clicks_cols:
     if c in self.df.columns:
      fig,ax=plt.subplots()
      sns.scatterplot(x=self.df[c],y=self.df['weekly_revenue'],ax=ax)
      ax.set_title(c+' vs revenue')
      plt.tight_layout()
      plt.show()

  def impr_vs_rev(self):
    for c in self.impr_cols:
     fig,ax=plt.subplots()
     sns.scatterplot(x=self.df[c],y=self.df['weekly_revenue'],ax=ax)
     ax.set_title(c+' vs revenue')
     plt.tight_layout()
     plt.show()

  def rev_vs_env(self):
    for f in self.other_cols:
     fig, ax = plt.subplots()
     sns.scatterplot(x=self.df[f],y=self.df['weekly_revenue'],ax=ax)
     ax.set_title(f + ' vs revenue')
     plt.tight_layout()
     plt.show()

  def corrmap(self):
    numz=self.df.select_dtypes(include='number')
    corr=numz.corr()
    fig,ax=plt.subplots(figsize=(13,10))
    sns.heatmap(corr,cmap='coolwarm',annot=False)
    ax.set_title('correlation matrix')
    plt.tight_layout()
    plt.show()

  def top_spend_chart(self):
    s = self.df[self.media_cols].sum().sort_values(ascending=0)
    fig,ax=plt.subplots()
    s.plot(kind='bar',ax=ax)
    ax.set_title('total spend per media')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
  
  def weekly_spend_trend_online(self):
    online_cols= [c for c in self.media_cols if any(k in c for k in ['facebook','instagram','youtube','search','display','influencer'])]
    fig,ax=plt.subplots(figsize=(12,6))
    self.df.set_index('week')[online_cols].plot(ax=ax)
    ax.set_title('online media spend over time')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

  def weekly_spend_trend_offline(self):
    offline_cols = [c for c in self.media_cols if c not in [x for x in self.media_cols if any(k in x for k in ['facebook','instagram','youtube','search','display','influencer'])]]
    fig,ax=plt.subplots(figsize=(12,6))
    self.df.set_index('week')[offline_cols].plot(ax=ax)
    ax.set_title('offline media spend over time')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
