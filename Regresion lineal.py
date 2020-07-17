import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as seabornInstance

datafile = 'datosRL.xls'

data = pd.read_excel(datafile, header=1)
print(data)
data2=data.transpose()
data2.dropna(axis=0,inplace=True)
data2.rename(columns=data2.iloc[0],inplace=True)
data2.drop(['Año'],inplace=True)
print(data2)



print(data2.describe())

data2.plot(x='consumo', y='Renta Neta', style='o')
plt.title('Renta Neta vs Consumo')
plt.xlabel('Consumo')
plt.ylabel('Renta Neta')

plt.show()

consumo_list = data2['consumo'].to_list()
renta_list= data2['Renta Neta'].to_list()

x = np.array(consumo_list).reshape((-1,1))
y = renta_list

model = LinearRegression()
model.fit(x,y)

r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)

Y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, Y_pred, color='red')


plt.show()
