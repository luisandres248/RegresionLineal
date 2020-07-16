import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

datafile = 'datosRL.xls'

data = pd.read_excel(datafile, header=1)
print(data)
data2=data.transpose()
data2.dropna(axis=0,inplace=True)
data2.rename(columns=data2.iloc[0],inplace=True)
data2.drop(['Año'],inplace=True)
print(data2)



print(data2.describe())


