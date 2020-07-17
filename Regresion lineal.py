import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.metrics import mean_squared_error

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

print('beta = ' + str(model.coef_) + ', alpha = ' + str(model.intercept_))

# Predecimos los valores y para los datos usados en el entrenamiento
prediccion_entrenamiento = model.predict(x)

# Calculamos el Error Cuadrático Medio (MSE = Mean Squared Error)
mse = mean_squared_error(y_true = y, y_pred = prediccion_entrenamiento)

# La raíz cuadrada del MSE es el RMSE
rmse = np.sqrt(mse)

print('Error Cuadrático Medio (MSE) = ' + str(mse))
print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))


Y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, Y_pred, color='red')


plt.show()
