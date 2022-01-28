#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Habilitar intellisense
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')
get_ipython().run_line_magic('autosave', '60')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


import pandas as pd
import numpy as np
import os
from scipy import stats


# In[3]:


ruta_archivo = os.path.join("titanic", "train.csv")
df = pd.read_csv(os.path.join("titanic", "train.csv"), index_col='PassengerId')


# In[4]:


df.head()


# ## 1 Elimine las variables/columnas 'Ticket' y 'Cabin'

# In[6]:


df.drop(columns=['Cabin'], inplace=True)
df.drop(columns=['Ticket'], inplace=True)
df.head()


# In[15]:


df.describe(include='all')


# ## 2 Encuentre los mejores valores para completar la variable 'Embarked' para los  pasajeros con datos faltantes

# In[93]:


df[df.Embarked.isnull()]


# In[46]:


#Tome en cuenta que simplemente emplear el valor que más aparece casi nunca es la mejor alternativa
#Para obtener mejores resultados deberá apoyarse en los resultados de un análisis exploratorio de datos
#aplicado a las variable(s) que puedan estar relacionadas con 'Embarked'
df['Embarked'].value_counts()


# In[91]:


# Obtenemos el puerto más común según la clase y el sexo
df.groupby(['Pclass', 'Sex'])['Embarked'].agg(pd.Series.mode)


# In[92]:


# Como podemos observar, todos los valores que se obtuvieron en la celda de arriba pertenecen al puerto 'S'
# Entonces, reemplazaremos los valores NaN por el puerto 'S'
df['Embarked'].fillna('S', inplace=True)
df[df.Embarked.isnull()]


# ## 3 Complete el código de la siguiente función para extraer el título de cada pasajero

# In[22]:


def extractTitle(name):
    title_mapping = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    title = name.split(', ')[1].split('.')[0].lower()
    return title_mapping[title]


# In[23]:


df['Title'] =  df['Name'].map(lambda name : extractTitle(name))
df.head()


# In[24]:


df.describe(include='all')


# ## 4 Encuentre los mejores valores para completar la variable 'Age' para los pasajeros con datos faltantes

# In[40]:


#Tome en cuenta que simplemente emplear el valor de tendencia central (mediana) de todo el conjunto casi nunca es la mejor alternativa
#Para obtener mejores resultados deberá apoyarse en los resultados de un análisis exploratorio de datos
#aplicado a las variable(s) que puedan estar relacionadas con 'Age'
df[df['Age'].isnull()].head()


# In[41]:


df[df['Age'].isnull()].shape


# In[26]:


df['Age'].median()


# In[28]:


df.corr()


# Según la matriz de correlación, *Age* tiene la correlación más alta con *Pclass*. Entonces, agruparemos los datos según este parámetro. También los agruparemos según el sexo, o sea, que los datos estarán agrupados por clase y por sexo. Porque las edades de las personas pueden variar según su sexo.

# In[29]:


pasajeros_por_clase_y_sexo = df.groupby(['Pclass', 'Sex'])
mediana_age_pasajeros = pasajeros_por_clase_y_sexo['Age'].transform('median')


# In[30]:


# Reemplazar los datos nulos de Age
df['Age'].fillna(mediana_age_pasajeros, inplace=True)

df[df['Age'].isnull()].head()


# ## 5 Cree una nueva variable 'IsMother' (1=es madre, 0=no es madre) 
# 
# ## Responda la siguiente pregunta sustentando su respuesta con los resultados de un análisis exploratorio de datos.
# ### Las madres tuvieron mayor probabilidad de sobrevivir al accidente del Titanic?

# Elegimos a las madres como las mujeres que están casadas y que viajen al menos con un Parch(parientes que sean padres o hijos).
# 
# No incluímos a las mujeres con el título de Miss, para evitar incluir a las mujeres que no son madres pero que 
# viajan con sus papás.
# 
# Hay que tener en cuenta que en esa época no habían muchas madres solteras (por el machismo), así que no se
# pierde mucha información en caso de que una mujer sea madre y no esté casada.

# In[40]:


def es_mujer_y_no_soltera_y_viaja_con_hijos(fila):
    return (fila['Parch'] > 0) & (fila['Sex'] == 'female') & (fila['Title'] != 'Miss')

def es_madre(fila):
    if es_mujer_y_no_soltera_y_viaja_con_hijos(fila):
        return 1
    return 0

df['IsMother'] = df.apply(es_madre, axis=1)
df.tail()


# In[41]:


df.loc[(df['IsMother'] == 1)]

