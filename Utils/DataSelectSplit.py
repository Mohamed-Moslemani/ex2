import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 

def XYSplit(df,x,y):

    X= df.drop(y).values 
    y= df[y].values 

def testtrainSplit(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=17)
    return x_train,x_test,y_train,y_test 


