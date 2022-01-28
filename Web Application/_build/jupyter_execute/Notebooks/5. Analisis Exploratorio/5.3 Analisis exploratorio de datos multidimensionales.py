#!/usr/bin/env python
# coding: utf-8

# # Análisis exploratorio de datos multivariable
# 
# 
# ## Objetivo
# 
# Identificar las correlaciones que existen en las variables para así poder formular hipótesis sobre relaciones causa-efecto.
# 
# ## Técnicas de análisis
# 
# 1. [Covarianza](#1)
# 2. [Coeficiente de correlación](#2)
# 3. [Visualización de la relación entre dos variables](#3)
# 4. [Variables cualitativas](#4)
#     1. [Crosstabs](#5)
#     2. [Pivot table (Tablas de contingencia)](#6)

# ## Carga de datos
# __[Fuente de datos](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)__

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
tips.tail()


# ### Covarianza
# <a id="1"></a>
# 
# Mide la relación lineal entre dos variables (si incrementan o decrementan juntas). Una covarianza positiva indica que las variables se mueven en la misma dirección y un covarianza negativa que las variables cambian en direcciones opuestas. Sin embargo, **no permite valorar la intensidad de la relación.**
# 
# <img src="./img/14-covarianza.png" style="width:600px"/>

# In[2]:


np.cov(tips['total_bill'], tips['tip'])


# In[3]:


tips.cov()


# ### Coeficiente de correlación 
# <a id="2"></a>
# 
# Al igual que la covarianza mide la dirección pero tambien la intensidad de la relación. Es importante recordar que  __[correlación no implica causalidad](https://www.gaussianos.com/hay-que-decirlo-mas-correlacion-implica-causalidad/)__
# 
# <img src="./img/15-correlacion.png" style="width:600px"/>
# 
# <img src="./img/16-rango-correlacion.png" style="width:600px"/>
# 
# Un coeficiente de correlación igual a cero no necesarimente implica que las variables sean independientes. Dos variables independientes tendrán un coefiente de correlación de cero. Por esta razón, **el cálculo del coeficiente de correlación debe ser complementado con la exploración visual de los datos.**. 
# 
# Tambien es importante recordar que el índice de correlación ***sólo indica relaciones lineales entre las variables***. Por tanto, no indica las relaciones no lineales que pudieran existir entre las variables. Por ejemplo, en la ecuación y=x^2, las variables tendrán una correlación de 0 a pesar de estar claramente relacionadas.
# 
# <img src="./img/17-correlacion-cero-no-implica-independencia.png" style="width:600px"/>

# In[4]:


np.corrcoef(tips['total_bill'], tips['tip'])


# In[5]:


tips.corr()


# ## Visualización de la relación entre dos variables
# 
# <a id="3"></a>
# 
# __[Seaborn](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)__
# 
# __[matplotlib](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.axvline.html?highlight=axvline)__

# In[6]:


tips.plot.scatter(x='total_bill', y='tip', color='c', title='scatter plot : Tip by Total bill', alpha=0.1)


# In[7]:


sns.jointplot(x='total_bill', y='tip', data=tips, color='#F15B2A')


# In[8]:


sns.jointplot(x='total_bill', y='tip', data=tips, color='#F15B2A', kind='kde')


# ## Variables cualitativas
# 
# <a id="4" ></a>
# 
# ### Crosstabs
# 
# <a id="5" ></a>

# In[9]:


tips.head()


# In[10]:


#Conteos de la combinacion de variables nominales
pd.crosstab(tips["smoker"], tips["sex"])


# In[11]:


pd.crosstab(tips["smoker"], tips["sex"]).plot(kind='bar')


# ### Pivot (tablas de contingencia)
# 
# <a id="6" ></a>
# 
# Es una extensión del crosstab que permite analizar dos variables categóricas y una numérica.
# 
# <img src="./img/18_pivot_table.png" style="width:600px"/>

# In[12]:


tips.pivot_table(index='time',columns = 'smoker',values='tip', aggfunc='mean')


# In[13]:


tips.groupby(['time','smoker'])['tip'].mean()
#type(tips.groupby(['time','smoker'])['tip'].mean())


# In[14]:


tips.groupby(['time','smoker'])['tip'].mean().unstack()


# In[15]:


tips.groupby(['time','smoker'])['tip'].mean().unstack().plot.bar()

