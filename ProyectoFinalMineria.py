import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("C:/Users/gamez/OneDrive/Escritorio/heart.csv")

#AGE: Edad de los pacientes
#Sex: Sexo del paciente; 0 = Femenino, 1 = Masculino
#Exang: Angina inducida por el ejercicio (1 = SI, 0 = NO)
#ca: Numero de vasos mayores (0.3)
#cp: Tipo de dolor
     #Valores: 1.Angina tipica, 2.Angina atipica, 3.Dolor no anginoso, 4.Asintomatico
#chol: Colesterol
#target: 0 = menos posibilidades de infarto, 1 = mas posibilidades de infarto
#trtbps : presión arterial en reposo
#thalachh : frecuencia cardíaca máxima alcanzada
#oldpeak: Depresión del ST (el descenso del segmento es un signo de daño miocardico) 
          #inducida por el ejercicio en relación con el reposo
     
#|||||||||||||||       DATA CLEANING        ||||||||||||||
dataset.isnull().sum()
dataset_borrar = dataset.dropna()
print(dataset_borrar)
dataset.duplicated().sum()
dataset.drop_duplicates(inplace=True)

#|||||||||||||||       DESCRIPTIVE STATISTICS      ||||||||||||||
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

#|||||||||||||||      DATA PARSING        |||||||||||||||||
#||||||||||||||      DATA VISUALIZATION    ||||||||||||||||
plt.figure(figsize=(18,10))
plt.style.use("ggplot")
sns.countplot(x = dataset["age"])   
plt.title("CONTEO DE LA EDAD DE LOS PACIENTES",fontsize=20)
plt.xlabel("EDAD",fontsize=20)
plt.ylabel("CONTADOR",fontsize=20)
plt.show()

c = dataset["cp"].value_counts().reset_index()
c
plt.figure(figsize=(20,10))
plt.style.use("ggplot")
sns.barplot(x = c["index"],y = c["cp"])   
plt.title("TIPO DE DOLOR MAS COMUN",fontsize=20)
plt.xlabel("TIPO",fontsize=20)
plt.ylabel("CONTADOR",fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.style.use("ggplot")
sns.displot(dataset["trtbps"])   
plt.title("DISTRIBUCION DE LA PRESION ARTERIAL DE LOS PACIENTES",fontsize=18)
plt.xlabel("PRESION ARTERIAL",fontsize=20)
plt.ylabel("CONTADOR",fontsize=20)
plt.show()

plt.figure(figsize=(20,10))
plt.style.use("ggplot")
sns.displot(dataset["chol"])
plt.title("DISTRIBUCION DEL NIVEL DE COLESTEROL",fontsize=18)
plt.xlabel("NIVEL DE COLESTEROL",fontsize=20)
plt.ylabel("CONTADOR",fontsize=20)
plt.show()

#|||||||||||        STATISTICS TEST          |||||||||||||||||
continous_cols = ['age', 'trtbps', 'chol','thalachh', 'oldpeak'] 
cnt = 0
max_in_row = 1
for x in continous_cols:
    data = dataset[x]
    plt.figure(cnt//max_in_row, figsize=(25,8))
    plt.subplot(1, max_in_row, (cnt)%max_in_row + 1)
    plt.title(f'Distribucion de {x} variable', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x, fontsize=16)
    plt.ylabel('Count', fontsize=16)
    sns.histplot(data, bins = 50, kde=50)
    plt.show()
    cnt += 1 
# Conclusiones de las pruebas de estadistica
print("\n")
print("CONLUSIONES DE LAS PRUEBAS DE ESTADISTICA")
print("- La mayoría de la gente sufre un ataque al corazón a los 50 años.")
print("- Las personas con mayor colesterol tienen menos probabilidades de sufrir un infarto.")
print("- Las personas con una frecuencia cardíaca máxima más alta tienen más probabilidades de sufrir un infarto.")

#|||||||||||        LINEAR MODELS              ||||||||||||||||
plt.figure(figsize = (20,10))
sns.regplot(x=dataset['age'],y=dataset['oldpeak'])
plt.show()

#|||||||||||        DATA CLASSIFICATION        ||||||||||||||||
plt.figure(figsize=(12,8))
plt.style.use("ggplot")
sns.histplot(data = dataset, x = 'age', hue = 'output')
plt.title("¿LA EDAD AFECTA A LOS ATAQUES DEL CORAZON?")
plt.xlabel("EDAD")
plt.ylabel("CONTADOR")
plt.show()

plt.figure(figsize=(18,8))
plt.title("COLESTEROL EN HOMBRES Y MUJERES")
sns.swarmplot(y="chol",x="output",data=dataset)
plt.xlabel("SEXO")
plt.ylabel("COLESTEROL")
plt.show()

#||||||||||||       DATA CLUSTERING         ||||||||||
figura = plt.figure(figsize=(15,5))
plantilla = figura.add_gridspec(1,2)
ax0 = figura.add_subplot(plantilla[0,0])
background_color = '#f6f5f7'
figura.patch.set_facecolor(background_color) 
ax0.scatter(x='oldpeak',y='thalachh',data=dataset[dataset['output']==0],alpha=0.5,color='purple',label = 'No hay ataque al corazon')
ax0.scatter(x='oldpeak',y='thalachh',data=dataset[dataset['output']==1],color='green',alpha=0.7,label = 'Ataque al corazon')
ax0.set_ylabel('Frecuencia Cardiaca Maxima')
ax0.set_xlabel('Depresion de ST')
ax0.legend()
ax0.set_title('Frecuencia Cardiaca Maxima Y Oldpeak',fontweight='bold')
plt.show()    

#||||||||||      FORECASTING      |||||||||||
X = dataset[['age', 'trtbps', 'chol', 'thalachh']]
y = dataset[['cp']]
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
