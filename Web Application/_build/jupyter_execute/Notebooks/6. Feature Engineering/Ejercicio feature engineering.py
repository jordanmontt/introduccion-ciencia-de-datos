#!/usr/bin/env python
# coding: utf-8

# # Ejercicio Feature Engineering
# 
# ### Reemplace todos los valores nulos o vacíos del archivo csv pertinente
# 
# Este ejercicio consiste en cargar un archivo csv que tiene valores nulos en más de una columna. Es requerido reemplazar los valores nulos por el mejor valor posible.

# 1. [Importar las librerias y cargar el archivo csv](#1)
# 2. [Visualizar que datos son los que faltan](#2)
# 3. [Ver los filas de los datos faltantes y la matriz de correlación](#3)
# 4. [Reemplazar los valores nulos de la columna mpg](#4)
# 5. [(Opcional) Reemplazar los valores nulos de mpg utilizando funciones programadas por mí](#5)
# 6. [Reemplazar los valores nulos de la columna caballos_potencia](#6)

# ## 1. Importar las librerias y cargar el archivo csv
# 
# <a id="1"></a>

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

autos = pd.read_csv(os.path.join('./csv/', 'datos_automoviles.csv'))


# ## 2. Visualizar que datos son los que faltan
# 
# <a id="2"></a>

# In[2]:


autos.describe(include='all')


# ## 3. Ver los filas de los datos faltantes y la matriz de correlación
# 
# <a id="3"></a>

# In[3]:


autos.loc[autos['mpg'].isnull()]


# In[4]:


autos.loc[autos['caballos_potencia'].isnull()]


# Se quiere saber que variable(columna) tiene la mejor correlación con la variable que tiene valores nulos
# en este caso, la variable que tiene la mejor correlación con mpg es peso.
# 
# Pero, como peso puede adqurir múltiples valores es mejor utilizar a la variable cilindros,
# porque cilindros sólo puede adqurir pocos valores, por tanto, el valor de la mediana será más preciso.

# In[5]:


autos.corr()


# ## 4. Reemplazar los valores nulos de la columna mpg
# 
# <a id="4"></a>

# In[6]:


cilindros_agrupacion = autos.groupby('cilindros')

# Esto crea una serie del mismo tamaño del DataFrame que contiene el valor de la mediana de las mpg con respecto
# a los cilindros.
mediana_mpg_por_cilindro = cilindros_agrupacion['mpg'].transform('median')

# Reemplazar los valores nulos de mpg con la mediana obteniad anteriormente
autos['mpg'].fillna(mediana_mpg_por_cilindro, inplace=True)


# ## 5. (Opcional) Reemplazar los valores nulos de mpg utilizando funciones programadas por mí
# 
# <a id="5"></a>

# In[7]:


cilindros_agrupacion = autos.groupby('cilindros')
# Crear una serie que tenga como índices los cilindros y como valor la mediana de las mpg seǵun los cilindros
mediana_mpg_segun_cilindros = cilindros_agrupacion['mpg'].median()

# Funcion que recibe una fila y si tiene la columna mpg como nan le pone el valor de la mediana de las mpg
# según el cilindro
def poner_mediana_en_mpg_nulos(fila):
    if math.isnan(fila['mpg']):
        fila['mpg'] = mediana_mpg_segun_cilindros[fila['cilindros']]
    return fila

# Aplicar  a cada fila del data frame la funcion definida arriba
autos_con_mpg = autos.apply(poner_mediana_en_mpg_nulos, axis=1)

# La misma funcion pero utilizando cálculo lambda
#autos_con_mpg = autos.apply(lambda x: mediana_mpg_segun_cilindros[x['cilindros']] if math.isnan(x['mpg']) else x, axis=1)


# ## 6. Reemplazar los valores nulos de la columna caballos_potencia
# 
# <a id="6"></a>

# In[8]:


# El procedimiento es el mismo que el de reemplazar los valores nulos de mpg
mediana_caballos_potencia_segun_cilindros = cilindros_agrupacion['caballos_potencia'].transform('median')
autos['caballos_potencia'].fillna(mediana_caballos_potencia_segun_cilindros, inplace=True)

