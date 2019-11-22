# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:25:39 2018

@author: Niloy
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_visualization(data):
    print (data.head())
    count=0
    for i in data:
        plt.subplot(int(len(data.columns)/3), 3, count+1)
        print(i)
        plt.hist(data[i])
        plt.title("Histogram for "+i, fontsize=8)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        count=count+1
    plt.show()


def plot_corr(df):
    print(df.columns)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.show()


def data_preprocessing(data):
    print("Missing values",data.isnull().sum())
    print(data.info())
    data.fillna(data.mean(), inplace=True)
    print(data.info())
    Y= data["Outcome"]
    X= data.drop("Outcome", axis=1)
    sc= StandardScaler()
    #sc= MinMaxScaler()
    X= sc.fit_transform(X)
    X=pd.DataFrame(X)
    print(X.head())
    #print( X.shape)
    X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=0,stratify=data['Outcome'])
    return X_train,X_test,y_train, y_test


def RFClassification(X_train,X_test,y_train, y_test):
    RFC= RandomForestClassifier(n_estimators=50, random_state=0)
    RFC= RFC.fit(X_train,y_train)
    y_pred= RFC.predict(X_test)
    return y_pred


def feature_selection(data):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    X = data[data.columns[:8]]
    Y = data['Outcome']
    model.fit(X, Y)
    print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))









