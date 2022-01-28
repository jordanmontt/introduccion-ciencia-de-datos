#!/usr/bin/env python
# coding: utf-8

# <a href="https://visit.figure-eight.com/rs/416-ZBE-142/images/CrowdFlower_DataScienceReport.pdf" target="_blank">
# DATA SCIENTIST REPORT 2017 - Adquisición y limpieza de datos consume la mayor parte del tiempo en un proyecto</a></br>

# In[1]:


#1 Analisis Exploratorio de Datos (EDA)
#Estructura básica (nro filas, nro de columnas, tipo de dato de las columnas, explorar el head y tail)
#  filas/columnas = observaciones/variables
#Comprender los datos con estadísticas básicas (categorico/nominal, ordinal, contínuo), 
#calidad de datos: datos faltantes, valores atípicos

#2 Preparación de datos (Data Wrangling/Data Munging)
#Reemplazar, eliminar datos faltantes, valores atípicos, realizar sumarizaciones, re-shaping

#Herramentas
#Numpy: módulos para manipulación de arreglos y matrices
#Pandas: manipulación de tablas (DataFrames)


# <a href="https://www.numpy.org/" target="_blank">Numpy</a></br>
# <a href="https://pandas.pydata.org/" target="_blank">Python Data Analysis Library (Pandas)</a></br>
# <a href="https://docs.python.org/3/library/functions.html#open" target="_blank">Abrir un archivo</a></br>
# <a href="http://elornitorrincoenmascarado.blogspot.com/2006/10/python-25-la-sentencia-with.html" target="_blank">With - Gestores de contexto</a></br>

# In[2]:


import requests
csv_url = "https://www.datos.gov.co/api/views/6zi7-wwa7/rows.csv?accessType=DOWNLOAD"
#respuesta = requests.get(csv_url)
#with open("archivo.csv", "wb") as archivo:
    #archivo.write(respuesta.content)


# In[3]:


import pandas as pd
import numpy as np
import os


# In[4]:


ruta_archivo = os.path.join("titanic", "train.csv")
print(ruta_archivo)
df = pd.read_csv(ruta_archivo, index_col='PassengerId')


# In[5]:


type(df)


# In[6]:


#info() muestra un resumen del DataFrame
#Pclass 
#SibSp numero hermanos , espos@   
#Parch numero padres , hij@s)
#Embarked punto de embarque C=Cherburgo Q=Queenstown S= Southampton
df.info()


# In[7]:


df.head(10)


# In[8]:


df.tail(10)


# In[9]:


#Concetenar DFs
#df = pd.concat((df_01, df_02),axis=0)
#Agregar columnas/variables
df["NuevaVariable"] = "Test"
df.head(5)


# In[10]:


#del df["NuevaVariable"]
df.head(5)


# In[11]:


#Igual que df.Name
df["Name"]


# In[12]:


type(df.Fare)


# In[13]:


df[['Name','Age']].head(5)


# In[14]:


#Seleccionar por filas y columnas
df.loc[7:10, ['Name','Age']]


# In[15]:


df.loc[0:5, "Name":"Age"]


# In[16]:


df.iloc[0:5, 2:5]


# In[17]:


#Filtrado
varones = df.loc[df.Sex == 'male',:]
print(type(varones))
varones.head(5)


# In[18]:


varones_en_primera_clase = df.loc[((df.Sex == 'male') & (df.Pclass == 1)),:]
varones_en_primera_clase.head(5)


# In[19]:


#Estadisticas
#Numerico: centralidad (media,mediana) y dispersion (rango, percentiles y desviacion estandar)
#Categorico: conteo, valores unicos, proporciones

#Numerico:
#centralidad
#  Media/Promedio, sensible a valores atípicos
#  Mediana
#dispersion (que tan lejos están de la tendencia central da una idea de la variabilidad de los dato)
#  rango: max - min, sensible a valores atípicos
#  percentiles: percentile X es Y, el X% de los valores son menores a Y
#    valores usuales para X 25, 50, 75 Quartiles
#  desviacion estandar: distancia de los valores con la media, sensible a valores atípicos
df.describe()


# In[20]:


print('Promedio del precio  : {0}'.format(df.Fare.mean()))
print('Mediana del precio : {0}'.format(df.Fare.median()))


# In[21]:


#Nan porque hay valores faltantes
print('Min precio : {0}'.format(df.Fare.min())) 
print('Max precio : {0}'.format(df.Fare.max()))
print('Rango precio : {0}'.format(df.Fare.max()  - df.Fare.min()))
print('25 percentil : {0}'.format(df.Fare.quantile(.25)))
print('50 percentil : {0}'.format(df.Fare.quantile(.5)))
print('75 percentil : {0}'.format(df.Fare.quantile(.75)))
print('Desviacion estandar precio : {0}'.format(df.Fare.std()))


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


#gráfico box-whisker
df.Fare.plot(kind='box')


# In[24]:


#Categorico:
#centralidad
#  Media/Promedio, sensible a valores atípicos
#  Mediana
df.describe(include='all')


# In[25]:


df.Sex.value_counts()


# In[26]:


df.Sex.value_counts(normalize=True)


# In[27]:


#Cuántos sobrevivieron? (Ejercicio)
df.Pclass.value_counts()


# In[28]:


df.Pclass.value_counts().plot(kind='bar')


# In[29]:


df.Pclass.value_counts().plot(kind='bar',rot = 0, title='Conteo de pasajes por clase', color='c');

