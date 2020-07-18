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

#archivo de datos
datafile = 'datosRL.xls'

#sacando datos del archivo hacia un dataframe
data = pd.read_excel(datafile, header=1)
print(data)
data2=data.transpose()
data2.dropna(axis=0,inplace=True)
data2.rename(columns=data2.iloc[0],inplace=True)
data2.drop(['Año'],inplace=True)
print(data2)
print(data2.describe())

#datos del dataframe a 2 listas
consumo_list = data2['consumo'].to_list()
renta_list= data2['Renta Neta'].to_list()

#modelo de regresion lineal
x = np.array(renta_list).reshape((-1,1))
y = consumo_list
model = LinearRegression()
model.fit(x,y)

#Ploteando Diagrama de dispersion
plt.scatter(x, y)
plt.title('Diagrama de dispersión')
plt.xlabel('Renta Neta')
plt.ylabel('Consumo')
plt.show()


#coeficiente de determinacion y coeficientes de recta
r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)
print('beta = ' + str(model.coef_) + ', alpha = ' + str(model.intercept_))


# Predecimos los valores y para los datos usados en el entrenamiento
prediccion_entrenamiento = model.predict(x)
print(prediccion_entrenamiento)
print(type(prediccion_entrenamiento))
y2 = np.array(y)
print(y2)
print(type(y2))

# Calculamos el Error Cuadrático Medio (MSE = Mean Squared Error)
mse = mean_squared_error(y_true = y2, y_pred = prediccion_entrenamiento)

# La raíz cuadrada del MSE es el RMSE
rmse = np.sqrt(mse)

print('Error Cuadrático Medio (MSE) = ' + str(mse))
print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))

#ploteo de regresion lineal
def plot_model(x,y):
    Y_pred = model.predict(x)
    plt.scatter(x, y)
    plt.plot(x, Y_pred, color='red')
    plt.title('Diagrama de dispersión Con Modelo de Regresión')
    plt.xlabel('Renta Neta')
    plt.ylabel('Consumo')
    plt.show()
plot_model(x,y)

#Calculo de consumo dado renta
rentax=480
consumoy=model.predict(np.array(rentax).reshape(-1,1))
print('consumo calculado con el modelo dado una renta de ' + str(rentax) + ' = ' + str(float(consumoy)))

renta_list.append(rentax)
consumo_list.append(int(consumoy))
print(renta_list)
print(consumo_list)

#ploteo de dispersion con dato nuevo y modelo de regresion
plt.xlim( 350, 600)
plt.ylim(250 , 450)
plot_model(np.array(renta_list).reshape((-1,1)),consumo_list)