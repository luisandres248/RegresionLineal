import scipy
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches
import seaborn as sns

def func(j):
    return stats.norm(mu1, (Sigma / np.sqrt(n))).isf(j)


# datos principales
x_ralla = 5.25
n = 16
mu1 = 5.5
Sigma = np.sqrt(3)
alpha = 0.05
mu, sigma0 = 0, 1  # media y desvio estandar

# calculo del estadistico normalizando la media muestral
estadistico = np.sqrt(n) * (x_ralla - mu1) / Sigma
print(estadistico)

# calculo de valor de Z para nivel de significancia 5%
scipy.stats.norm(0, 1)
z_alpha = stats.norm(0, 1).ppf(alpha)
print(z_alpha)
# calculo de x ralla critico con nivel de significancia 5%
s2 = np.random.normal(mu1, Sigma, 1000)  # creando muestra de dator para plotear despues con datos muestrales mu1
X_ralla_critico = stats.norm(mu1, (Sigma / np.sqrt(n))).ppf(0.05)
print(X_ralla_critico)

# segunda parte
mu2 = 5.4

# calculo de potencia del test dado mu2 como verdadero
errortipo2 = stats.norm(mu2, (Sigma / np.sqrt(n))).sf(X_ralla_critico)
print(errortipo2)
potencia = 1 - errortipo2
print(potencia)

# calculo de pvalor
pvalor = stats.norm(0, 1).pdf(estadistico)
print(pvalor)

# plotting graphs

xmuestral_mu1 = np.linspace(mu1 - 5 * Sigma, mu1 + 5 * Sigma, 100)
plt.plot(xmuestral_mu1, stats.norm.pdf(xmuestral_mu1, mu1, Sigma), label='X muestral con mu1')
xmuestral_mu2 = np.linspace(mu2 - 5 * Sigma, mu2 + 5 * Sigma, 100)
plt.plot(xmuestral_mu2, stats.norm.pdf(xmuestral_mu2, mu2, Sigma), label='X muestral con mu2')


plt.legend()

plt.title('Funcion de probabilidad acumulada - Distribucion normal')

plt.show()
