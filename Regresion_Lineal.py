import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

USAhousing = pd.read_csv("C:/Users/gamez/OneDrive/Escritorio/USA_Housing.csv")

#'Avg. Area Income': Ingresos de los residentes de la ciudad en la que se encuentra la casa.
#'Avg. Area House Age': Edad media de la casa en la ciudad.
#'Avg. Area Number of Rooms': Número medio de habitaciones de las casas de la misma ciudad.
#'Avg. Area Number of Bedrooms': Número medio de dormitorios de las casas de la misma ciudad.
#'Area Population': Población de la ciudad en la que se encuentra la casa.
#'Price': Precio al que se vendio la casa
#'Address': direccion de la casa

print(USAhousing.head())
print(USAhousing.info())

#Dvidimos los datos en una matriz X.
#Una matruz Y con la variable del destino, en este caso la columna precio.
#No ocupamos laculumna Direccion ya que solo tiene texto y en la regresion no se puede usar.
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)


# Predicciones del conjunto de datos
print(lin_reg.intercept_)

pred = lin_reg.predict(X_test)

plt.scatter(y_test, pred)
plt.show()

