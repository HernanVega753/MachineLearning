import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('simple_data/housing.csv/housing.csv')
print(data.head())

data.info()

print(data['ocean_proximity'].value_counts())

data.hist(bins=50, figsize=(20, 15))
plt.show()

data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()

data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population'] / 100, label='population',
          figsize=(15, 7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.show()

# Correlación
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(15, 10))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.show()

corr_matrix = numeric_data.corr()

print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Combinación de atributos
# -rooms_per_household: Representa el numero de habitaciones por hogar
# en una cierta área. Proporciona una mediad de la densidad de habitaciones
# en una vivienda promedio en esa área.

# -bedrooms_per_room: indica la proporción de dormitorios con respecto al
# número total de habitaciones en una cierta área

# -population_per_household: representa la desidad de población promedio por hogar en una cierta área.

numeric_data['rooms_per_household'] = data['total_rooms'] / data['households']
numeric_data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
numeric_data['population_per_household'] = data['population'] / data['households']

corr_matrix = numeric_data.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# Limpieza de datos y manejar atributos categóricos

data.info()
x = [1, 2, 3, np.nan]
x1 = pd.Series(x)
print(x1.mean())
# da 2
x = [1, 2, 3, 0]
x1 = pd.Series(x)
print(x1.mean())
# da 1.5
# Poner ceros afecta el valor final de la columna, debe usarse el promedio o la mediana
x = [1, 2, 3, 2]
x1 = pd.Series(x)
print(x1.mean())
# da 2

# Normalización de datos con mediana
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())

print(data.info())

# Manipulación de los datos categóricos
data_ocean = data[['ocean_proximity']]
ordinal_encoder = OrdinalEncoder()
data_ocean_encoder = ordinal_encoder.fit_transform(data_ocean)

print(np.random.choice(data_ocean_encoder.ravel(), size=10))

# Trasformar a contraparte numérica
cat_encoder = OneHotEncoder()
data_car_1hot = cat_encoder.fit_transform(data_ocean)

print(data_car_1hot.toarray())

print(cat_encoder.categories_)

encoded_df = pd.DataFrame(data_car_1hot.toarray(), columns=cat_encoder.get_feature_names_out())
print(encoded_df.head())

# Agregar las columnas calculadas al DataFrame original
data['rooms_per_household'] = numeric_data['rooms_per_household']
data['bedrooms_per_room'] = numeric_data['bedrooms_per_room']
data['population_per_household'] = numeric_data['population_per_household']

# -------------------------------------------------------------------------

# ALGORITMOS DE MACHINE LEARNING

# Variables dependientes e independientes
y = data['median_house_value'].values.reshape(-1, 1)

# Selección de columnas para X
X = data[[
    'median_income',
    'rooms_per_household',
    'total_rooms',
    'housing_median_age',
    'households'
]]

# Concatenar las variables independientes con los datos codificados
data1 = pd.concat([X, encoded_df], axis=1)

# Verificar las columnas del DataFrame final
print(data1.columns)

X = data1.values

# Algoritmo de regresión lineal múltiple
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)

print(data1.head)

# Escalar variables
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(r2)

columnas = [
    'median_income',
    'rooms_per_household',
    'total_rooms',
    'housing_median_age',
    'households',
    'latitude',
    'longitude'
]
col_modelo = []
y = data['median_house_value'].values.reshape(-1, 1)

for col in columnas:
    col_modelo.append(col)
    data1 = data[col_modelo]
    data1 = pd.concat([data1, encoded_df], axis=1)
    X = data1.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    y_pred = lin_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print('Columnas: ', col_modelo, r2)

    #---------------------------------------------------------------------------------------
    