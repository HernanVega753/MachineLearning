import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# MAQUINA DE SOPORTE VECTORIAL - Es para datos COMPLEJOS
data = pd.read_csv('simple_data/Advertising.csv')
data = data.iloc[:, 1:]

# TV NEWSPAPER
x = data.drop(['Radio', 'Sales'], axis=1).values
y = data['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svr = SVR(kernel='rbf')  # rvf es el default
svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)


print('Reales:', y_test[:4], 'Predicción:', y_pred)
print(r2_score(y_test, y_pred))

# TV RADIO
# TV NEWSPAPER
x = data.drop(['Newspaper', 'Sales'], axis=1).values
y = data['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svr = SVR(kernel='rbf')  # rvf es el default
svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)


print('Reales:', y_test[:4], 'Predicción:', y_pred)
print(r2_score(y_test, y_pred))




