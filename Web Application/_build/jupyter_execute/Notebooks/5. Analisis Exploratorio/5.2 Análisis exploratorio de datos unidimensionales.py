#!/usr/bin/env python
# coding: utf-8

# # Análisis unidimensional
#  <a id="1"></a>
# 
# ## Objetivo
# 
# Como ya se dijo anteriormente, el análisis unidimensional consiste en analizar individualmente a las variables (columnas de un DataFrame) para conocer sus características y su naturaleza. Para ello, se emplea mayormente gráficos, aunque tambien se pueden calcular valores estadísticos como el promdio, la mediana, la kurtosis, entre muchos otros.
# 
# ## Ejemplo de como graficar
# 
# - [Ejemplo visualización](#8)
# 
# ## Técnicas de análisis
# 
# 1. [DataFrame.describe()](#3)
# 2. [Indicadores de tendencia central](#1)
# 3. [Indicadores de dispersión](#2)
# 4. [Gráficos de distribución](#4)
# 5. [Gráficos de comparación](#5)
# 6. [Gráficos de composición](#6)
# 7. [Utilizando Groupby](#7)

# #### Importar lbrerías y cargar datos
# __[Fuente de los datos](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)__

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargarlos desde la web
data = pd.read_csv('http://www.dsi.uminho.pt/~pcortez/forestfires/forestfires.csv')

data.head()


# ## DataFrame.describe()
# <a id="3"></a>
# 
# Llamando a la función`describe()` de _Pandas_ se obtiene una matriz la cual tiene varios indicadores para cada una de las columnas. Con estos indicaroes se puede tener una visión general de los datos.

# In[2]:


data_description = data.describe(include='all')
data_description


# ## Visualización
# <a id="8"></a>
# 
# Esto es solamente un ejemplo de la manipulación de las librerías para graficar. En la celda de abajo, se grafica la distribución de los valores de la *temperatura* y se añaden líneas al gráfico las cuales muestran los indicares de tendencia central así como los límites de los valores atípicos.
# 
# Para graficar los datos se utilizan estas dos librerías:
# 
# - [Seaborn](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
# 
# - [matplotlib](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axvline.html?highlight=axvline)
# 
# 
# 

# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
sns.kdeplot(data['temp'], shade=True)

# Agrega lineas verticales en los indicadores de la tendencia central
plt.axvline(data['temp'].mean(), color='g') # Agrega una línea color verde, la cual indica el promedio
plt.axvline(data['temp'].median(), color='black') # Agrega una línea color negro, la cual indica la mediana
plt.axvline(data_description['temp']['25%'], color='black') # Agrega una línea color negro, la cual indica el Q1
plt.axvline(data_description['temp']['75%'], color='black') # Agrega una línea color negro, la cual indica el Q3

IQR = data_description['temp']['75%'] - data_description['temp']['25%']

upper_outliers = data_description['temp']['75%'] + 1.5*IQR
lower_outliers = data_description['temp']['25%'] - 1.5*IQR

# Agrega dos líneas de color rojo, las cuales indican los límites para los valores atípicos
plt.axvline(upper_outliers, color='r') 
plt.axvline(lower_outliers, color='r')
plt.show()


# ## Indicadores de tendencia central 
# <a id="1"></a>
# 
# Los indicadores que son sirven para medir la tendencia central son:
# 
# - **Media**

# In[4]:


from statistics import mean

a1 = data['temp'].mean() # utilizando la librería pandas
a2 = mean(data['temp'])  # utilizando la librería  statistics
a3 = np.mean(data['temp']) # utilizando la librería  numpy

print(f'{a1}, {a2}, {a3}')


# - **Mediana**

# In[5]:


from statistics import median

a1 = data['temp'].median() # pandas
a2 = median(data['temp'])  # statistics
a3 = np.median(data['temp']) # numpy

print(f'{a1}, {a2}, {a3}')


# - **Cuantiles**
# 
# Son valores que dividen a los datos en cuatro partes iguales.
# 
#     - 1er cuantil (Q1) 	25% de los datos es menor que o igual a este valor.
#     - 2do cuantil (Q2) 	La mediana. 50% de los datos es menor que o igual a este valor.
#     - 3er cuantil (Q3) 	75% de los datos es menor que o igual a este valor. 
# 
# <img src="./img/02-quartiles.png" style="width:600px"/>

# ## Indicadores de dispersión
# <a id="2"></a>
# 
# Estos indicares muestran cuan dispersos están los datos.
# 
# - **Varianza, desviación estándar**
# 
# <img src="./img/03-dispersion.png" style="width:600px"/>

# In[6]:


from statistics import stdev
from statistics import variance

std1 = data['temp'].std() # pandas
std2 = stdev(data['temp'])  # statistics
std3 = np.std(data['temp']) # numpy

varianza1 = data['temp'].var() # pandas
varianza2 = variance(data['temp'])  # statistics
varianza3 = np.var(data['temp']) # numpy

print(f'Desviación estándar: {std1}, {std2}, {std3}')
print(f'Varianza: {varianza1}, {varianza2}, {varianza3}')


# - **Valores atípicos**
# 
# <img src="./img/04-outliers.png" style="width:600px"/>

# - **Asimetría (skweness). Grado de simetría de la distribución.**
# 
# <img src="./img/05-skewness.png" style="width:600px"/>

# In[7]:


from scipy.stats import skew

a1 = data['temp'].skew() # pandas
a2 = skew(data['temp'])  # scipy

print(a1, a2)


# - **Curtosis. Indicador de la "anchura" de una distribución**
# 
# <img src="./img/06-kurtosis.png" style="width:600px"/>

# In[8]:


from scipy.stats import kurtosis

a1 = data['temp'].kurt() # pandas
a2 = kurtosis(data['temp'])  # scipy

print(a1, a2)


# ## Gráficos de distribución
# <a id="4"></a>
# 
# Para visualizar los gráficos, se puede utilizar, en vez de librerias externas como en el ejemplo de arriba, la librería *pandas*. __[Panda's Plots](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html)__ 
# 
# - **[Histograma](https://es.wikipedia.org/wiki/Histograma)**
# 
# Permite visualizar las frecuencias de diferentes categorías o rangos de valores (clases o bins)
# 
# <img src="./img/07-histograma.png" style="width:400px"/>
# 
# 

# In[9]:


data['temp'].plot.hist(title='Histograma Temp', color='c', bins=20)


# - __[Diagrama de Densidad](https://es.wikipedia.org/wiki/Funci%C3%B3n_de_densidad_de_probabilidad)__
# 
# <img src="./img/08-density-plot.png" style="width:400px"/>

# In[10]:


data['temp'].plot.density(title='Densidad Temp', color='c')


# - __[Diagrama de Caja](https://es.wikipedia.org/wiki/Diagrama_de_caja)__ 
# 
# Este es especialmente útil para la identificación de valores atípicos.
# 
# <img src="./img/09-box-plot.png" style="width:400px"/>

# In[11]:


data['temp'].plot.box(title='Box Temp', color='c')


# ## Gráficos de comparación
# <a id="5"></a>
# 
# __[Panda's Plots](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html)__ 
# 
# - __[Gráfico de Barras](https://es.wikipedia.org/wiki/Diagrama_de_barras)__ 
# 
# Presenta datos **cualitativos** con barras rectangulares con alturas o longitudes proporcionales a los valores que representan. (No son histogramas!!)
# 
# <img src="./img/10-bar-chart.png" style="width:400px"/>

# In[12]:


df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
df.plot.bar(x='lab', y='val', rot=0)


# - __[Gráfico de lineas](https://en.wikipedia.org/wiki/Line_chart)__
# 
# Presenta una serie de valores (marcas) conectados por líneas. Se emplean frecuentemente para visualizar tendencias a lo largo del tiempo.
# 
# <img src="./img/11-line-chart.png" style="width:400px"/>
# 
# Para ambos tipos de gráficos, se recomienda aplicarlos para un número reducido de variables (~ 6)

# ## Gráficos de composición
# <a id="6"></a>
# 
# __[Panda's Plots](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html)__ 
# 
# - __[Torta](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html)__ 
# 
# <img src="./img/12-pie-chart.png" style="width:400px"/>
# 

# In[13]:


df = pd.DataFrame({'masa': [0.330, 4.87 , 5.97],
                   'radio': [2439.7, 6051.8, 6378.1]},
                  index=['Mercurio', 'Venus', 'Tierra'])

# Las dos maneras son equivalentes
df.plot.pie(y='masa', figsize=(5, 5))
df["masa"].plot.pie()


# - __[Barras apiladas](https://businessq-software.com/2017/02/21/stacked-bar-chart-definition-and-examples-businessq/)__
# 
# <img src="./img/13-stacked-bar.png" style="width:400px"/>
# 

# In[14]:


df = pd.DataFrame({'vestimenta': [8261.68, 7875.87 , 4990.23],
                   'equipamiento': [4810.34, 3126.58, 4923.48],
                   'accesorios': [1536.57, 2019.81, 1472.59],},
                  index=['Cherry St', 'Strawberry Mall', 'Peach St'])
df


# In[15]:


df.plot.bar(stacked=True, rot=0)


# ## Utilizando Groupby
# <a id="7"></a>
# 
# __[Panda's groupby](https://pandas.pydata.org/pandas-docs/version/0.23.1/api.html#groupby)__ 
# 
# __[Panda's DataFrameGroupBy aggr](https://pandas.pydata.org/pandas-docs/version/0.23.1/generated/pandas.core.groupby.DataFrameGroupBy.agg.html)__
# 
# La función groupby puede ser beneficiosa. Por ejemplo, si queremos saber qué temperatura promedio tiene cada mes se hace lo siguiente:

# In[16]:


# Este celda agrega una columna month_number la cual tiene el número del mes.
# Por ejemplo, para una fila que tiene jan como mes, el valor de month_number será 1.
   
def month_number(month):
    if 'jan' == month:
        return 1
    elif 'feb' == month:
        return 2
    elif 'mar' == month:
        return 3
    elif 'apr' == month:
        return 4
    elif 'may' == month:
        return 5
    elif 'jun' == month:
        return 6
    elif 'jul' == month:
        return 7
    elif 'aug' == month:
        return 8
    elif 'sep' == month:
        return 9
    elif 'oct' == month:
        return 10
    elif 'nov' == month:
        return 11
    elif 'dec' == month:
        return 12
    return 0

data["month_number"] = data["month"].map(month_number)


# In[17]:


# Ahora, agruparemos los datos según la nueva columna month_number
temperaturas_por_mes = data.groupby("month_number")

# Finalmente, sacaremos el promedio de las temperaturas por mes
temperaturas_por_mes.agg({'temp' : 'mean'})

