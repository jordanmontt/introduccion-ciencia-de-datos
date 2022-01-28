#!/usr/bin/env python
# coding: utf-8

# # Ejercicio análisi exploratorio parte 1
# 
# Utilizando los datos sobre automoviles extraídos en un archivo csv via webscraping (están almacenados en un archivo csv en la misma ruta que este notebook), responda a las siguientes preguntas:
# 
# 1. [La asimetría de la variable `caballos_potencia` es negativa?](#1)
# 2. [Entre las variables: `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, ¿cuáles tienen valores atípicos en ambos extremos?](#2)
# 3. [Entre las variables: `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, ¿cuáles no tienen valores atípicos?](#3)
# 4. [Entre las variables: `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`,, ¿cuáles son las variables con mayor y menor asimetría?](#4)
# 5. [Entre las variables: `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, muestre los valores atípicos de aquellas variables que los tengan en ambos extremos.](#5)

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

automoviles = pd.read_csv('./csv/datos_automoviles.csv')
automoviles.head()


# ### 1. La asimetría de la variable `caballos_potencia` es negativa?
# 
# <a id="1"></a>

# In[2]:


print('Asimetría de caballos_potencia')
automoviles['caballos_potencia'].skew()


# ### 2. Entre las variables: `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, ¿cuáles tienen valores atípicos en ambos extremos?
# 
# <a id="2"></a>
# 
# Para saber cúales variables tiene valores atípicos, se puede utilizar gráficos o métido el analítico. En este caso, se utilizarán gráficas de dos tipos:
# 
# - Una será una gráfica de distribución donde se dibujarán líneas verticales a la altura de ambos valores atípicos.
# - La otra, un *gráfico de caja* el cual nos señala mediante un círculo si existen o no valores atípicos
# 
# Cabe resaltar que la primera manera no indica de manera precisa la existencia de valores atípicos, pero se lo muestra por propósitos ilustrativos. En cambio, el *gráfico de caja* si indica con exactidud si existen o no valores atípicos.

# In[3]:


def graficar_distribucion_con_valores_atipicos(columna):
    q1 = automoviles.describe()[columna]['25%']
    q3 = automoviles.describe()[columna]['75%']
    iqr = q3 - q1
    limite_derecho = q3 + 1.5 * iqr
    limite_izquierdo = q1 - 1.5 * iqr

    sns.kdeplot(automoviles[columna], shade=True)
    plt.axvline( limite_derecho, color='b')
    plt.axvline( limite_izquierdo, color='b')


# #### 2.1 Valores atípicos `caballos_potencia`

# In[4]:


graficar_distribucion_con_valores_atipicos('caballos_potencia')


# In[5]:


automoviles['caballos_potencia'].plot.box()


# #### 2.2 Valores atípicos `desplazamiento`

# In[6]:


graficar_distribucion_con_valores_atipicos('desplazamiento')


# In[7]:


automoviles['desplazamiento'].plot.box()


# #### 2.3 Valores atípicos `mpg`

# In[8]:


graficar_distribucion_con_valores_atipicos('mpg')


# In[9]:


automoviles['mpg'].plot.box()


# #### 2.4 Valores atípicos `aceleracion`

# In[10]:


graficar_distribucion_con_valores_atipicos('aceleracion')


# In[11]:


automoviles['aceleracion'].plot.box()


# #### Respuesta pregunta 2: la `aceleración` es la única variable que tiene valores atípicos en ambos extremos

# ### 3. Entre las variables:  `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, ¿cuáles no tienen valores atípicos?
# 
# <a id="3"></a>
# 
# Sobre la base de los gráficos hechos en el anterior inciso, `desplazamiento` es la variable que no tiene valores atípicos

# ### 4. Entre las variables:  `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, ¿cuáles son las variables con mayor y menor asimetría?
# 
# <a id="4"></a>

# In[12]:


print('caballos_potencia: ', automoviles['caballos_potencia'].skew())
print('desplazamiento: ', automoviles['desplazamiento'].skew())
print('mpg: ', automoviles['mpg'].skew())
print('aceleracion:', automoviles['aceleracion'].skew())


# ### 5. Entre las variables:  `caballos_potencia`, `desplazamiento`, `mpg` y  `aceleracion`, muestre los valores atípicos de aquellas variables que los tengan en ambos extremos
# 
# <a id="5"></a>
# 
# Como se pudo observar en el __[inciso 2](#2)__ la variable `aceleracion` es la única que tiene valores atípicos en ambos extremos.

# In[13]:


q1 = automoviles.describe()['aceleracion']['25%']
q3 = automoviles.describe()['aceleracion']['75%']
iqr = q3 - q1
limite_derecho = q3 + 1.5 * iqr
limite_izquierdo = q1 - 1.5 * iqr


# Valores atípicos menores o iguales al límite inferior:

# In[14]:


automoviles.loc[automoviles['aceleracion'] <= limite_izquierdo]


# Valores atípicos mayores o iguales al límite superior:

# In[15]:


automoviles.loc[automoviles['aceleracion'] >= limite_derecho]


# - Hay alguna relacion entre el territorio y los caballos de potencia?
# - Hay correlación entre los caballos de potencia y las millas por galón? 
# - Si existe es un relación, ambas variables se mueven en la misma dirección?
# - Compruebe la dirección de la relación con una visualización.

# In[16]:


def territorio_escrito_a_num(territorio):
   return territorio.map({ 'Europe': 1, 'USA': 2, 'Japan': 3 })

automoviles['territorio_num'] = territorio_escrito_a_num(automoviles['territorio'])
automoviles


# In[17]:


automoviles.corr()


# Los caballos de potención tienen una relación con las millas por galón, debido el valor de correlación es 0.77 y es próximo a -1. Lo cual indica una relación inversa.

# In[18]:



automoviles.plot.scatter(x='caballos_potencia', y='mpg', color='c', title='scatter plot : Tip by Total bill', alpha=0.1)


# No se mueven en la misma dirección

# In[19]:


automoviles.cov()

