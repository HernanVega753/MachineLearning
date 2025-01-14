#  REGRESIÓN LINEAL POLINÓMICA

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # transformación polinómica
from sklearn.metrics import r2_score

pos = [x for x in range(1, 11)]
post = ["Pasante de Desarrollo",
        "Desarrollador Junior",
        "Desarrollador Intermedio",
        "Desarrollador Senior",
        "Lider de Proyecto",
        "Gerente de Proyecto",
        "Arquitecto de Software",
        "Director de Desarrollo",
        "Director de Tecnología",
        "Director Ejecutivo (CEO)"]
salary = [1200.0, 2500.0, 4000.0, 4800.0, 6500.0, 9000.0, 12820.0, 15000.0, 25000.0, 50000.0]

data = {
    'position': post,
    'years': pos,
    'salary': salary
}
data = pd.DataFrame(data)

print(data.head())

plt.scatter(data['years'], data['salary'])
plt.show()

x = data.iloc[:, 1].values.reshape(-1, 1)
y = data.iloc[:, -1].values

regression = LinearRegression()
regression.fit(x, y)

plt.scatter(data['years'], data['salary'])
plt.plot(x, regression.predict(x), color="black")
plt.show()

print(regression.predict([[2.5]]))

#  MODELO POLINÓMICO

poly = PolynomialFeatures(degree=10)  # Instancia
x_poly = poly.fit_transform(x)
print(x_poly)

regression_2 = LinearRegression()
regression_2.fit(x_poly,y)
plt.scatter(data['years'], data['salary'])
plt.plot(x, regression_2.predict(x_poly), color="black")

plt.show()

predict = poly.fit_transform([[2]])
print(regression_2.predict(predict))

y_pred = regression_2.predict(x_poly)
print(r2_score(y, y_pred))
