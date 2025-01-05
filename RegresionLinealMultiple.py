#  REGRESIÓN LINEAL MÚLTIPLE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # para transformar los arrays de Python en arrays de numpy
import seaborn as sns

data = pd.read_csv('simple_data/Advertising.csv')

X = data.drop(['Newspaper', 'Sales'], axis=1).values
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

#  Gráfico en Seaborn
sns.regplot(x=y_test, y=y_pred)


plt.show()
