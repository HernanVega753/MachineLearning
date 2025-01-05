# Regresión Lineal

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # para transformar los arrays de Python en arrays de numpy

data = pd.read_csv('simple_data/Advertising.csv')
data.head()
#print((data.head()))

data = data.iloc[:,1:]  # Conjunto de datos que NO utilizar, desde cual(0) hasta cual(1)

#print((data.head()))

#data.info()  # Da información general del dataset
print(data.describe())  # Da un resumen general estadístico de datos numéricos

print(data.columns)  # Vista de columnas del dataset

cols = ['TV', 'Radio', 'Newspaper']

for col in cols:
    plt.plot(data[col], data['Sales'], 'ro')
    plt.title('Ventas respecto a la publicidad en %s' % col)
    plt.show()

X = data['TV'].values.reshape(-1, 1)
print(X)
y = data['Sales'].values

# Dividir entre entrenamiento y testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # 20% para testing, definimos el random

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print(X_train.shape)  # Vemos la cantidad con la que se entrenará nuestro modelo
print(X_test.shape)  # Vemos la cantidad de comprobaciones

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

print('Predicciones: {}, REALES: {}'.format(y_pred[:4], y_test))

# RMSE

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)
#R2
print('R2: ', r2_score(y_test, y_pred))

plt.plot(X_test, y_test, 'ro')
plt.plot(X_test, y_pred)
plt.show()


def modelos_simple(independiente):
    X = data[independiente].values.reshape(-1, 1)
    y = data['Sales'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # 20% para testing, definimos el random
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)

    print('Predicciones: {}, REALES: {}'.format(y_pred[:4], y_test))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: ', rmse)
    # R2
    print('R2: ', r2_score(y_test, y_pred))

    plt.plot(X_test, y_test, 'ro')
    plt.plot(X_test, y_pred)
    plt.show()


modelos_simple('Radio')
modelos_simple('TV')
modelos_simple('Newspaper')
