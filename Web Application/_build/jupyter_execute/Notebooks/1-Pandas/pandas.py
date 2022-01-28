#!/usr/bin/env python
# coding: utf-8

# # Pandas
# __[Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)__ es un paquete construido sobre la base de NumPy, incluye la implementación de la estructura **DataFrame**. Un DataFrame es, en esencia, un arreglo bidimensional con etiquetas para filas y columnas, típicamente las columnas contienen tipo de datos diferentes.

# 1. [Series](#1)
# 2. [DataFrame](#2)
# 3. [Índices](#3)
# 4. [Obtención de datos](#4)
# 5. [Se utiliza *(&, |)* en lugar de *(and, or)*](#8)
# 6. [Modificación de datos](#5)
# 7. [Apply](#6)
# 8. [One Hot Encoding](#7)

# In[1]:


import numpy as np
import pandas as pd


# ## Series
# <a id="1"></a>
# Un objecto de tipo *Series* es un arreglo de datos, parecido a un *Array* de *numpy*, que consta de índices y valores.
# 
# Aquí algunos enlaces de referencia:
# - https://pandas.pydata.org/pandas-docs/stable/reference/series.html#computations-descriptive-stats
# - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html

#  **Una serie tiene varios métodos, como `min, mean, std`, entre muchos otros. Para crear una serie:**

# In[10]:


serie = pd.Series([0.25, 0.5, 0.75, 1.0])
print(serie)
print('Desviación estándar: ', serie.std())


# **Una serie tiene valores e índices:**

# In[11]:


print('Valores: ', serie.values)
print('Índices: ', serie.index)


# **Filtrado de datos, retorna una Serie de valores booleanos:**

# In[12]:


serie > 0.5


# In[13]:


serie.isnull()


# **A diferencia de los arreglos de *Numpy*, a una Serie se le puede asignar un indice de manera explícta:**

# In[14]:


serie = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd']) 
print(serie['a':'c'])


# **Se puede crear una Serie a partir de un diccionario (clave -> indice)**

# In[15]:


poblacion_dict = {'Chuquisaca': 626000, 
                  'La Paz': 26448193,
                  'Cochabamba': 2883000,
                  'Oruro': 538000,
                  'Potosí': 887000,
                  'Tarija': 563000,
                  'Santa Cruz': 3225000,
                  'Beni': 468000,
                  'Pando': 144000 }
poblacion = pd.Series(poblacion_dict)
poblacion


# **Otros ejemplos de creación de Series**

# In[16]:


serie = pd.Series(5, index=[100, 200, 300])
serie


# **Selección de claves del diccionario (solo se crea un serie con una parte del diccionario)**

# In[17]:


serie = pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
serie


# ## Dataframes
# <a id="2"></a>
# Un *DataFrame* es un arreglo bi-dimensional formado por una secuencia de Series con la misma cantidad de elementos y con el mismo índice. Es decir: es como un diccionario de Series del mismo tamaño y con los mismos índices. Un *DataFrame* permite asignar nombres a las columnas.

# In[18]:


extension_departamentos_Bolivia_dict = {'Chuquisaca': 51514, 
                    'La Paz': 133985,
                    'Cochabamba': 55631,
                    'Oruro': 55588,
                    'Potosí': 117218,
                    'Tarija': 37623,
                    'Santa Cruz': 370621,
                    'Beni': 213564
                  }
extension_departamentos_Serie = pd.Series(extension_departamentos_Bolivia_dict)
extension_departamentos_Serie


# **Creación a partir de dos Series que tiene el mismo index (aunque los indices no estén en el mismo order o incluso falten datos en algunas de las Series)**

# In[19]:


datos_bolivia = pd.DataFrame({'poblacion': poblacion, 'extension': extension_departamentos_Serie})
datos_bolivia


# **Tanto las filas como las columnas tienen asociado un índice**

# In[20]:


print(datos_bolivia.index)
print(datos_bolivia.columns)


# **Se puede ver a un DataFrame como un diccionario de Series (columnas)**

# In[21]:


datos_bolivia['poblacion']


# **Otras maneras de crear un DataFrame: si no se provee un índice se crea una secuencia de numeros que empieza en 0.**

# In[22]:


data = pd.DataFrame(columns=['a','b'], data=[[1, 45], [87, 96], [125, 13], [135, 789]])
data


# **Lista de diccionarios (las claves son los nombres de las columnas)**

# In[23]:


data = pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
data


# **Información general de un *DataFrame***

# In[24]:


datos_bolivia.shape


# In[25]:


datos_bolivia.head(5)


# In[26]:


datos_bolivia.tail(5)


# In[27]:


datos_bolivia.size


# In[28]:


datos_bolivia.info()


# In[29]:


# la función describe() devuele un DataFrame con indicadores para cada una de las columnas
datos_bolivia.describe()


# # Indices
# <a id="3"></a>
# Un *Index* es el mecanismo para referenciar datos en las Series y los DataFrames. Un Index object es un **conjunto** ordenado de valores

# In[30]:


indA = pd.Index([1, 3, 5, 7, 9]) 
indB = pd.Index([2, 3, 5, 7, 11])


# In[31]:


print(indA.union(indB))
print(indA.intersection(indB))
print(indA.difference(indB))


# # Extracción de datos
# <a id="4"></a>
# 
# Extraer datos de un DataFrame o una serie.

# In[32]:


datos_bolivia = pd.DataFrame(data={'poblacion':poblacion, 'extension':extension_departamentos_Serie})
datos_bolivia


# **Un *DataFrame* es como diccionario de Series (columnas) en el cual se puede extraer y modificar datos**

# In[33]:


datos_bolivia['poblacion']
datos_bolivia[['poblacion','extension']]
datos_bolivia['constante'] = 1
datos_bolivia['densidad'] = datos_bolivia['poblacion'] / datos_bolivia['extension']
datos_bolivia


# In[46]:


datos_bolivia['capital'] =  pd.Series({'Chuquisaca': 'Sucre', 
                    'La Paz': 'Murillo',
                    'Cochabamba': 'Cercado',
                    'Oruro': 'Cercado',
                    'Potosí': 'Potosí',
                    'Tarija': 'Tarija',
                    'Santa Cruz': 'Santa Cruz de la Sierra',
                   'Pando': 'Cobija',
                    'Beni': 'Trinidad' })
datos_bolivia


# **Un DataFrame es también como un arreglo bidimensional (una matriz de Series)
# Soporta indices, slicing, filtering empleando los indices explicitos (iloc usa indices numéricos implicitos).
# El primer valor de la matriz hace referencia a las filas**
# 
# - https://railsware.com/blog/python-for-machine-learning-indexing-and-slicing-for-lists-tuples-strings-and-other--sequential-types/

# In[35]:


datos_bolivia.loc['Beni']


# In[36]:


datos_bolivia.loc['Beni':'Oruro']


# In[37]:


datos_bolivia['poblacion'] > 2000000


# In[38]:


datos_bolivia['extension'].isnull()


# ## Se utiliza *(&, |)* en lugar de *(and, or)*
# <a id="8"></a>

# In[39]:


datos_bolivia.loc[(datos_bolivia['poblacion'] > 2000000) & (datos_bolivia['extension']> 60000.0), ['poblacion','densidad'] ]
datos_bolivia


# # Modificación de datos
# <a id="5"></a>

# **Elimina todos los datos de una columna**

# In[40]:


datos_bolivia.drop(columns=['constante'], inplace=True)
datos_bolivia


# **Eliminar los datos faltantes (los que son NaN)**

# In[41]:


datos_bolivia.dropna(how='any')


# In[42]:


datos_bolivia.loc['Pando', 'densidad'] = datos_bolivia.loc['Pando', 'poblacion'] / datos_bolivia.loc['Pando', 'extension']
datos_bolivia


# ## Apply
# <a id="6"></a>
# Appy aplica una función que recibe como argumento a cada una de las columnas (o filas) de un DataFrame. Modifica el DataFrame existente.

# **axis=0 es la opción por defecto, significa que se recorrerá el DataFrame por las columnas (similar a recorrer una matriz por columas). Si axis=1 el DataFrame se recorrerá por sus filas.**

# In[43]:


datos_bolivia_extension_reducida_a_la_mitad = datos_bolivia.apply(lambda x: x['extension']/2, axis=1)
datos_bolivia_extension_reducida_a_la_mitad


# ## One Hot Encoding
# <a id="7"></a>
# Conversión de valores numéricos y nominales en categorías y luego las categorías en valores numéricos.
# Necesario cuando el algoritmo de aprendizaje automático no es capaz de trabajar con valores nominales o contínuos

# **Obtener los códigos de una variable nominal**

# In[44]:


datos_bolivia['capital'].astype('category').cat.codes


# **Obtener el vector One Hot Encoding**

# In[45]:


pd.get_dummies(datos_bolivia,columns=['capital'])

