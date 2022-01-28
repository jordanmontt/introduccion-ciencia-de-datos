#!/usr/bin/env python
# coding: utf-8

# # Feature engineering (preparación de variables)

# 1. [Definicion](#1)
# 2. [Imputación](#2)
# 3. [Valores atípicos](#3)
# 4. [Binning](#4)
# 5. [Transformación logarítmica](#5)
# 6. [One-hot encoding](#6)
# 7. [Separación de valores](#7)
# 8. [Ajuste de escala](#8)

# ## Definición
# 
# <a id="1"></a>
# 
# __[What Is Feature Engineering](https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a)__
# 
# Proceso de aplicación del conocimiento de los datos de cierto ámbito/dominio para seleccionar o crear variables que mejoren el desempeño de los modelos predictivos. Se recomienda realizar luego del Análisis Exploratorio de Datos.
# 
# ## Técnicas
# 
# - Imputación, manejo de valors faltantes (eliminar o encontrar un valor adecuado)
# - Manejo de valores atípicos, eliminarlos o preservarlos.
# - Binning, agrupar valores en clases típicamente para convertir variables contínuas en discretas.
# - Transformación logaritmica, para lidiar con distribuciones muy asimétricas
# - One-hot enconding, convertir variables nominales en 0s y 1s
# - Separación de valor (Feature Split), ej convertir nombre completo en nombre y apellido.
# - __[Ajuste de escala](https://en.wikipedia.org/wiki/Feature_scaling)__., para ubicar variables en rangos recomendados
# 
# 
# <img src="./img/01-feature-engineering.png" style="width:600px"/>

# ![](./01-eda-visual-techniques.png)

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(os.path.join("./csv/diabetes.csv"))
df.head()


# ## Imputación
# 
# <a id="2"></a>

# In[2]:


#df.isnull()
df.describe(include='all')


# In[3]:


df.loc[3,'Age'] = np.nan
df.head()


# In[4]:


df.describe(include='all')


# In[5]:


#df['Age'].isnull()


# In[6]:


df.loc[df['Age'].isnull()]


# In[7]:


df.shape


# In[8]:


#Eliminación de valores faltantes
df.dropna(how='all').shape


# In[9]:


df.dropna(subset=['Insulin', 'Age'], how='any').shape


# In[10]:


df.head()


# In[11]:


df.dropna(subset=['Insulin', 'Age'], how='any', inplace=True )
df.head()


# In[12]:


#Asignación de valores
df = pd.read_csv(os.path.join("diabetes.csv"))
df.head()


# In[15]:


df.loc[3,'Age'] = np.nan


# In[16]:


df.head()


# In[17]:


#df['Age'].fillna(0, inplace=True) #Casi nunca es buena idea!
df['Age'].fillna(round(df['Age'].mean()), inplace=True) #Pocas veces es buena idea!
df.head()


# In[18]:


df = pd.read_csv(os.path.join("diabetes.csv"))
df.loc[3,'Age'] = np.nan


# In[19]:


df.head()
#df.shape


# In[20]:


#df.loc[df['Age'].notnull(),].head()


# In[21]:


por_embarazos = df.groupby('Pregnancies')
por_embarazos


# In[22]:


por_embarazos.groups


# #### Se recomienda emplear la métrica de tendencia central que sea menos afectada por valores atípicos:
# 
# **La Mediana.**

# In[24]:


#por_embarazos.agg({'Age': ['mean','median']})
por_embarazos.agg({'Age': 'median'})


# #### El gráfico de caja muestra la media o mediana?

# In[25]:


df[df['Age'].notnull()].boxplot('Age','Pregnancies')


# In[26]:


por_embarazos


# In[27]:


por_embarazos['Age']


# In[28]:


por_embarazos['Age'].transform('median')


# In[29]:


df['Age'].fillna(por_embarazos['Age'].transform('median'), inplace=True)
df.head()


# ## Valores atípicos
# 
# <a id="3"></a>

# In[30]:


df = pd.read_csv(os.path.join("diabetes.csv"))


# In[31]:


df['Age'].plot.box()


# In[32]:


# Identificación basada en percentiles (también existe la basada en la desviación estándar)
q3 = df['Age'].quantile(.75)
q1 = df['Age'].quantile(.25)

IQR = q3 - q1

df.loc[(df['Age'] > q3 + 1.5 * IQR) | (df['Age'] < q1 - 1.5 * IQR)]


# In[33]:


#df = df.loc[(df['Age'] <= q3 + 1.5 * IQR) & (df['Age'] >= q1 - 1.5 * IQR)]
df.loc[(df['Age'] <= q3 + 1.5 * IQR) & (df['Age'] >= q1 - 1.5 * IQR)].shape


# ## Binning
# 
# <a id="4"></a>

# ![](https://www.saedsayad.com/images/Binning_1.png)

# In[34]:


df = pd.read_csv(os.path.join("diabetes.csv"))


# In[35]:


df.describe()


# In[36]:


df['YoungAdult'] = df['Age'].map(lambda age: 1 if age <= 35 else 0 ) # age <= 35 ? 1 : 0
df.head()


# In[37]:


df.loc[df['YoungAdult'] == 0].shape


# In[38]:


df.loc[df['YoungAdult'] == 1].shape


# In[39]:


#df['BloodPressure_Bin'] = pd.qcut(df['BloodPressure'], 4, labels=['very_low','low','high','very_high'])
pd.qcut(df['BloodPressure'], 4, labels=['very_low','low','high','very_high'])


# In[40]:


df['AgeCategogy'] = pd.cut(df['Age'],bins=[0, 35, 55, 120], labels=['young', 'middle', 'old'])
df.head()


# ## Transformación logarítmica
# 
# <a id="5"></a>
# 
# Recuerde que log(0) = infinito

# In[41]:


df['Pregnancies'].plot.density(color='c')


# In[42]:


df['Pregnancies'].skew()


# In[43]:


np.log(df['Pregnancies'] + 1.0).plot.density(color='c')


# ## One-hot encoding
# 
# <a id="6"></a>
# 
# <img src="./img/02-one-hot-encoding.png" style="width:600px"/>

# In[44]:


df = pd.read_csv(os.path.join("diabetes.csv"))


# In[45]:


df['AgeCategogy'] = pd.cut(df['Age'],bins=[0, 35, 55, 120], labels=['young', 'middle', 'old'])
df.head()


# In[46]:


df = pd.get_dummies(df,columns=['AgeCategogy'])
df.head()


# ## Separación de valores
# 
# <a id="7"></a>

# In[47]:


df = pd.DataFrame({'Team':['Eagles', 'Bears', 'Raptors', 'Hornets', 'Bees', 'Lions'], 
                     'City':['Rome', 'Helsinki', 'Hong Kong', 'Hong Kong', 'Rome', 'Rome'],
                     'Games':[12, 15, 23, 18, 21, 8],
                     'MVP_Player': ['John Stuart', 'Leo Da Vinci', 'Mike Donatello', 'Raphael Dolce', 'Bruce Lee', 'Mahatma Gandhi']})
df.head()


# In[48]:


def extract_name(fullname):
    return fullname.split(' ')[0]


# In[49]:


#df['Name'] = df['MVP_Player'].apply(lambda fullname: fullname.split(' ')[0])
df['Name'] = df.apply(lambda row: row['MVP_Player'].split(' ')[0], axis = 1 )
df['Name'] = df['MVP_Player'].apply(extract_name)
df.head()


# ## Ajuste de escala
# 
# <a id="8"></a>
# 
# El ajuste de escala es una transformación aplicada a variables numéricas que tiene como objetivo asegurar que los valores de diferentes variables estén en el mismo rango. Esta transformación es necesaria cuando se emplean algoritmos sensibles a las magnitudes de las variables.
# 
# El método de ajuste más utilizado se basa en el cálculo del valor z (puntuación estándar, z-score); genera valores centrados en cero y con una desviación estándard igual a 1.
# 
# El valor Z mide las desviaciones estándar de distancia entre un valor y la media.
# 
# __[Boston house prices dataset](https://scikit-learn.org/stable/datasets/index.html#boston-dataset)__

# In[50]:


from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


# In[51]:


boston_dataset  = load_boston()
df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
df.head()


# In[52]:


scaler = StandardScaler()
scaler.fit(df)


# In[53]:


array = scaler.transform(df)
array


# In[54]:


df_scaled = pd.DataFrame(array, columns=df.columns)
df_scaled.head()


# In[55]:


#Revisión con menos datos
data = [[-1]
        , [-0.5]
        , [0]
        , [1]
       ]
scaler.fit(data)
scaler.transform(data)


# In[56]:


mean_a = np.array([-1,-0.5, 0, 1]).mean()
std_a = np.array([-1,-0.5, 0, 1]).std()


# In[57]:


print(mean_a)
print(std_a)


# In[58]:


(-1 - mean_a) / std_a


# In[59]:


(data[3][0] - mean_a) / std_a

