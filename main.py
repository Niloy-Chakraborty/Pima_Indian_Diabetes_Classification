# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:25:39 2018

@author: Niloy
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pima

'''
Function Name: main 
Description: main function for calling other functions
'''

def main():
    data= pd.read_csv("diabetes.csv")
    data1 = data[data["Outcome"]==1]
    pima.data_visualization(data1)

    pima.plot_corr(data[:8])
    pima.feature_selection(data)

    data= data[["Glucose","BMI","Age","DiabetesPedigreeFunction","Outcome"]]
    pima.plot_corr(data)
    print(data.shape)
    X_train,X_test,y_train, y_test= pima.data_preprocessing(data)
    y_pred = pima.RFClassification(X_train,X_test,y_train, y_test)
    cm = confusion_matrix(y_test, y_pred)
    print (cm)
    print(f1_score(y_test, y_pred))
    print("accuracy:", metrics.accuracy_score(y_test,y_pred))


if __name__ == "__main__":
    main()