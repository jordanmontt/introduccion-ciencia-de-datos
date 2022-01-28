#!/usr/bin/env python
# coding: utf-8

# # Ejercicio análisi exploratorio parte 2
# 
# Utilizando los mimos datos sobre automoviles, responda a las siguientes preguntas 
# 
# 1. [¿Existe alguna relacion entre el territorio y los caballos de potencia?](#1)
# 2. [¿Cuál es la correlación entre los caballos de potencia y las millas por galón?](#2)
# 3. [¿Si existiese una relación, ambas variables se mueven en la misma dirección?](#3)
# 4. [Compruebe la dirección de la relación con una visualización.](#4)

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

autos = pd.read_csv(os.path.join("./csv/", "datos_automoviles.csv"))
autos.head()


# ## 1. ¿Existe alguna relacion entre el territorio y los caballos de potencia?
# 
# <a id="1"></a>
# 
# `territorio` es una variable categórica, por tanto, no es posible calcular la correlación con `caballos_potencia`. Pero, como `territorio` solo puede tomar 3 valores (Europe, Japan, USA), convertir cada valor en un número y crear una nueva columna que tenga el `territorio` almacenado como un número. Así, sí es posible calcular la correlación.

# In[2]:


# Se crea la nueva columna
def territorio_a_num(territorio):
    serie_paises = ['Europe', 'USA', 'Japan']
    if territorio == serie_paises[0]:
        return 1
    elif territorio == serie_paises[1]:
        return 2
    elif territorio == serie_paises[2]:
        return 3
    
autos['territorio_num'] = autos['territorio'].map(territorio_a_num)

# La correlación entre las variables
autos.corr()['territorio_num']['caballos_potencia']


# ## 2. ¿Cuál es la correlación entre los caballos de potencia y las millas por galón?
# 
# <a id="2"></a>

# In[3]:


autos_corr = autos.corr()
autos_corr


# In[4]:


autos_corr['caballos_potencia']['mpg']


# ## 3. ¿Si existiese una relación, ambas variables se mueven en la misma dirección?
# 
# <a id="3"></a>
# 
# Como se puede apreciar, la covarianza es negativa, lo cual indica que las variables se mueven en direcciones contrarias.

# In[5]:


autos.cov()['mpg']['caballos_potencia']


# ## 4. Compruebe la dirección de la relación con una visualización
# 
# <a id="4"></a>
# 
# Las gráficas de regresión, se observa que la pendiente de la curva es negativa, lo cual indica que las variables son inversamente proporcionales, lo cual significa que las variables se mueven en direcciones contrarias.

# In[6]:


# Graficar utilizando seaborn
sns.jointplot(x='caballos_potencia', y='mpg', data=autos, color='#000000')

# Graficar utilizando pandas
autos.plot.scatter(x='caballos_potencia', y='mpg', color='#000000', title='scatter plot : mpg por caballos_potencia')

