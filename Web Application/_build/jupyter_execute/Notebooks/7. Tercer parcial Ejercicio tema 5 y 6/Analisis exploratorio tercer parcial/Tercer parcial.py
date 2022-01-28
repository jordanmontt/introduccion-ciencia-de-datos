#!/usr/bin/env python
# coding: utf-8

# # Tercer parcial
# 
# #### Alumnos:
# - Sebastian Jordan
# - Daniel Oropeza
# - Franz Vilela

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')
get_ipython().run_line_magic('autosave', '60')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Importar las librerias y cargar el archivo csv

# In[147]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

pasajeros_titanic = pd.read_csv(os.path.join('.', 'train.csv'))
pasajeros_titanic


# ## 2. Visualizar los indicadores y los datos faltantes

# In[148]:


pasajeros_titanic.describe(include='all')


# ## 3. Ver la matriz de correlación

# In[1]:


pasajeros_titanic.corr()


# ## 4. Reemplazar los datos faltantes

# ### 4.1 Reemplazar los datos de la columna "Fare"

# In[150]:


# Ver los datos faltantes
pasajeros_titanic.loc[pasajeros_titanic['Fare'].isnull()]


# Recordemos que las columnas *SipSp* y *Parch* representan el número de parientes (esposos, hermanos, padres, hijos), o sea, si los pasajeros están viajando solos o no y con cuantos acomañantes.
# 
# La correlación entre Fare y Pclass es la más alta de todas (-0.549500 ). Sólo falta una fila y es de un pasajero que viaja en tercera clase (Pclass = 3) y viaja solo (SibSp = 0 y Parch = 0).

# In[151]:


pasajeros_por_clase_y_por_numero_acompanantes = pasajeros_titanic.groupby(['Pclass', 'SibSp', 'Parch'])
mediana_fare_pasajeros = pasajeros_por_clase_y_por_numero_acompanantes['Fare'].transform('median')

# Reemplazar el dato faltante de Fare
pasajeros_titanic['Fare'].fillna(mediana_fare_pasajeros, inplace=True)


# ### 4.2 Reemplazar los datos de la columna "Embarked"

# In[152]:


pasajeros_titanic.loc[pasajeros_titanic['Embarked'].isnull()]


# ### 4.3 Reemplazar los datos de la columna "Age"

# In[155]:


pasajeros_titanic.loc[pasajeros_titanic['Age'].isnull()]


# Según la matriz de correlación, *Age* tiene la correlación más alta con *Pclass*. Entonces, agruparemos los datos según este parámetro. También los agruparemos según el sexo, o sea, que los datos estarán agrupados por clase y por sexo. Porque las edades de las personas pueden variar según su sexo.

# In[3]:


pasajeros_por_clase_y_sexo = pasajeros_titanic.groupby(['Pclass', 'Sex'])
mediana_age_pasajeros = pasajeros_por_clase_y_sexo['Age'].transform('median')

# Reemplazar los datos nulos de Age
pasajeros_titanic['Age'].fillna(mediana_age_pasajeros, inplace=True)


# ### 4.4 Reemplazar los datos de la columna "Cabin"

# In[156]:


pasajeros_titanic.loc[pasajeros_titanic['Cabin'].isnull()]


# In[221]:


pasajeros_sin_cabina = pasajeros_titanic.loc[pasajeros_titanic['Cabin'].isnull()]
pasajeros_con_cabina = pasajeros_titanic.loc[pasajeros_titanic['Cabin'].notnull()]

valores = [pasajeros_con_cabina.size, pasajeros_sin_cabina.size]
pd.Series(valores, index =["Con Cabina", "Sin Cabina"]).plot.pie(autopct='%1.1f%%')


# Como podemos observar, una gran mayoria de los datos (filas) no tiene la columna *Cabin*. Además que si observamos los datos, el valor de la cabina es una cadena que indica, probablemente, en qué cabina estaba el pasajero. Entonces es muy difícil determinar con qué valores se puede reemplazar los valores restantes. Por tanto, la mejor opción es eliminar la columna entera.

# In[215]:


# pasajeros_titanic.drop(columns=['Cabin'], inplace=True)

