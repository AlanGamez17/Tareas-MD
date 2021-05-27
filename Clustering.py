import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv("C:/Users/gamez/OneDrive/Escritorio/CCGENERAL.csv")

data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(),inplace=True)
data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)
data.isnull().sum()
data.drop('CUST_ID',axis=1,inplace=True)

from scipy.stats import zscore

data_scaled=data.apply(zscore)
data_scaled.head()

cluster_range = range(1,15)
cluster_errors=[]
for i in cluster_range:
    clusters=KMeans(i)
    clusters.fit(data_scaled)
    labels=clusters.labels_
    centroids=clusters.cluster_centers_,3
    cluster_errors.append(clusters.inertia_)
clusters_df=pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})
clusters_df

kmean= KMeans(4)
kmean.fit(data_scaled)
labels=kmean.labels_

clusters=pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()

clusters.groupby('cluster').mean()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf.head(2)

finalDf = pd.concat([principalDf, pd.DataFrame({'cluster':labels})], axis = 1)
finalDf.head()

plt.figure(figsize=(15,10))
ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="cluster", data=finalDf,palette=['red','blue','green','purple'])
plt.show()

print("\n")
print("Grupo 0 : Jóvenes / Nuevos profesionales en activo\n"
      "Saldo bajo, pero el saldo se actualiza con frecuencia, es decir, más número de transacciones.\n" 
      "El número de compras de la cuenta también es bastante grande y la mayoría de las compras se hacen de una sola vez o a plazos,\n" 
      "pero no pagando en efectivo por adelantado.")
print("\n")
print("Grupo 1 : Profesionales jubilados/pensionistas\n"
      "El saldo es relativamente alto, pero no se actualiza con frecuencia, es decir, hay menos transacciones.\n" 
      "El número de compras de la cuenta es bastante bajo y las compras de una sola vez o a plazos son muy escasas.\n" 
      "La mayoría de las compras se hacen pagando en efectivo por adelantado. La frecuencia de las compras también es bastante baja.\n")
print("\n")
print("Grupo 2: Industriales\n"
      "El saldo es muy alto y se actualiza con mucha frecuencia.\n" 
      "El número de compras es comparativamente menor y casi todas las compras se realizan en efectivo por adelantado.\n" 
      "La frecuencia de las compras también es bastante baja.\n")
print("\n")
print("Grupo 3: Empresarios de alto nivel\n"
      "El número de compras es extremadamente alto y la mayoría de sus compras se realizan de una sola vez o a plazos.\n" 
      "La frecuencia de compra también es muy alta, lo que indica que las compras se realizan con gran frecuencia.\n" 
      "Además, tienen el límite de crédito más alto.")