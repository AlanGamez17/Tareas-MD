import numpy as np 
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import init_notebook_mode
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#Para poder ver la informacion mas ordenada, si no se lo ponia se veia muy pegada
pd.set_option('display.max_columns', None)

dataset = pd.read_csv("C:/Users/gamez/OneDrive/Escritorio/netflix_titles.csv")

#Infromacion de los Datos
print("\n")
print("ESTADISTICAS BASICAS DEL DATAFRAME")
print(dataset.describe())
print("\n")
print("COLUMNAS DEL DATAFRAME")
print(dataset.columns)
print("\n")
print("TIPOS DE DATOS")
print(dataset.dtypes)
print("\n")
print("INDEX")
print(dataset.index)
print("\n")
print("DATAFRAME")
print("\n")
print(dataset)


# Diviendo los datos de años de lanzamiento para tener una mejor visualizacion
ry_40_80 = dataset[dataset["release_year"].isin(np.arange(1940, 1980 + 1 , 1))]
ry_81_20 = dataset[dataset["release_year"].isin(np.arange(1981, 2020 + 1 , 1))]
ry48 = ry_40_80["release_year"]
ry82 = ry_81_20["release_year"]

#Analisis del numero de caracteristicas del pais
print("\n")
ry = pd.Series(dataset["release_year"])
cu = pd.Series(dataset["country"])
year_country_data = pd.DataFrame({"Año_Realizacion":ry, "Ciudad":cu})
print(year_country_data.head())
print("\n")

#Mostrar si la Serie/Pelicula es de un pais en concreto
print("\n")
print("HECHAS EN INDIA")
print(dataset["country"].str.contains("India"))
print("\n")
print("HECHAS EN MEXICO")
print(dataset["country"].str.contains("Mexico"))
print("\n")
print("HECHAS EN ESTADOS UNIDOS")
print(dataset["country"].str.contains("United States"))

# Mostrando grafica de acuerdo a su año de lanzamiento 1940-1980
ry_40_80 = dataset["release_year"]
bins=np.arange(min(ry48), max(ry48) + 1, 1)
fig,ax = plt.subplots(figsize=(30,10))
ax.hist(ry_40_80, bins=bins, edgecolor='black', color='purple') # bins to spread the data year wise
ax.set_xticks(bins) # This will set the year exact with each bar
plt.xticks(rotation=90) # rotating the year vertically in y axis
ax.set(xlabel="Año de Lanzamiento",
      ylabel="Numero de Series/Peliculas")
ax.set_title("1940 a 1980", fontsize=20)
plt.show()

# Mostrando grafica de acuerdo a su año de lanzamiento 1981-2020
ry_81_20 = dataset["release_year"]
bins=np.arange(min(ry82), max(ry82) + 1, 1)
fig,ax = plt.subplots(figsize=(30,10))
ax.hist(ry_40_80, bins=bins, edgecolor='black', color='purple')
ax.set_xticks(bins)
ax.set(xlabel="Año de Lanzamiento",
      ylabel="Numero de Series/Peliculas")
ax.set_title("1981 a 2020", fontsize=20)
plt.xticks(rotation=90)
plt.show()



