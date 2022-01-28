#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Reshaping
# 
# Antes de analizar los datos, se necesita formar los datos obtenidos en un formato regular y que sea procesable por el algoritumo que luego utilizaremos. Es necesario asegurar que todos los datos correspondan con las variables. También, es necesario lidiar con los valores nulos, si es que hubiese. En términos generales, se puede decir que Data Reshaping es cambiar la manera en que los datos están organizados en coumnas y filas.

# ## Join 
# 
# Join, o merge, es el proceso de unir dos DataFrame diferentes en uno solo. Por ejemplo, si tenemos dos DataFrames que contienen diferente información pero sobre los mismos clientes, podemos unirlos en uno solo.
# 
# Aquí algunos enlaces extras:
# 
# __[pandas.merge](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.merge.html)__
# 
# __[Pandas : How to Merge Dataframes using Dataframe.merge() in Python – Part 1](https://thispointer.com/pandas-how-to-merge-dataframes-using-dataframe-merge-in-python-part-1/)__
# 
# __[Pandas : How to Merge Dataframes using Dataframe.merge() in Python – Part 2](https://thispointer.com/pandas-merge-dataframes-on-specific-columns-or-on-index-in-python-part-2/)__
# 
# __[Pandas : How to Merge Dataframes using Dataframe.merge() in Python – Part 3](https://thispointer.com/pandas-how-to-merge-dataframes-by-index-using-dataframe-merge-part-3/)__

# In[32]:


import pandas as pd
import numpy as np
import os


# In[33]:


mall_customers_info = pd.read_csv(os.path.join("csv", "mall_customers_info.csv"))
mall_customers_info.tail()


# In[34]:


mall_customers_info.shape


# In[35]:


mall_customers_score = pd.read_csv(os.path.join("csv", "mall_customers_score.csv"))
mall_customers_score.tail()


# In[64]:


mall_customers_score.shape


# In[63]:


#customer_data = pd.merge(mall_customers_info[['CustomerID','Gender','Annual_Income']], mall_customers_score, on='CustomerID')
customer_data = pd.merge(mall_customers_info,mall_customers_score,on='CustomerID')
customer_data.tail()


# In[39]:


customer_data.shape


# In[40]:


customer_data = pd.merge(mall_customers_info,mall_customers_score,on='CustomerID', how='left')
customer_data.tail()


# In[65]:


customer_data = pd.merge(mall_customers_info,mall_customers_score,on='CustomerID', how='right')
customer_data.tail()


# In[42]:


customer_data = pd.merge(mall_customers_info,mall_customers_score,on='CustomerID', how='outer')
customer_data.tail()


# In[43]:


customer_data = pd.merge(mall_customers_info,mall_customers_score,on='CustomerID', how='inner')
customer_data.tail()


# In[44]:


customer_data.shape


# ## Union
# __[pandas.concat](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.concat.html)__

# In[48]:


mall_customers_more = pd.read_csv(os.path.join("csv", "customers_data_2.csv"))
mall_customers_more.head()


# In[49]:


mall_customers_more.shape


# In[50]:


customers_data_all = pd.concat([customer_data, mall_customers_more])
#customers_data_all.sample(10)
customers_data_all.tail(10)


# In[51]:


customers_data_all.shape


# In[52]:


customers_data_all.reset_index(inplace=True, drop=True)


# In[53]:


#customers_data_all.sample(10)
customers_data_all.tail(10)


# ## stack, unstack
# 
# __[Reshape using Stack() and unstack() function in Pandas python](http://www.datasciencemadesimple.com/reshape-using-stack-unstack-function-pandas-python/)__

# In[54]:


datos_mensuales = pd.read_csv('./csv/monthly_data.csv')
datos_mensuales.head(5)


# In[23]:


# Preparacion: usar 'YYYY' como el ID/Indice
datos_mensuales.set_index('YYYY', inplace=True)
datos_mensuales.head(5)


# In[24]:


#En valor en cada columna se transforma en una fila
datos_mensuales.stack()


# In[55]:


athletes = pd.read_csv('./csv/athletes.csv')
athletes.info()


# In[26]:


athletes.head(5)


# In[27]:


weight_mean_by_sport_and_sex = athletes.groupby(['sport', 'sex'])['weight'].mean()
weight_mean_by_sport_and_sex


# In[28]:


#Mueve cada valor del ultimo nivel de un indice mutinivel a un columna
weight_mean_by_sport_and_sex.unstack()


# ## Pivot
# 
# __[pandas.DataFrame.pivot](https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.pivot.html)__

# In[56]:


#El ID se repite para cada uno de las propiedaded del producto :/
products = pd.DataFrame({'id': [823905, 823905,
                         235897, 235897, 235897,
                         983422, 983422],
                  'item': ['prize', 'unit', 
                           'prize', 'unit', 'stock', 
                           'prize', 'stock'],
                  'value': [3.49, 'kg',
                            12.89, 'l', 50,
                            0.49, 4]})
products


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
# 
# De manera similar a unstack, sirve para mover fila como columnas
# y así construir un DF con más columnas y menos filas
# 
# Usar id como el indice de cada fila y
# los valores en la columna 'item' para crear columnas.
# Los valores que debe aparecer en cada columnas esta en la columna 'value

# In[30]:


products.pivot(index='id', columns='item', values='value')


# In[61]:


stocks = pd.read_csv('https://gist.githubusercontent.com/alexdebrie/b3f40efc3dd7664df5a20f5eee85e854/raw/ee3e6feccba2464cbbc2e185fb17961c53d2a7f5/stocks.csv')
stocks.head(10)


# In[62]:


stocks.pivot(index='symbol', columns='date', values='volume')


# ## Melt
# 
# __[pandas.DataFrame.melt](https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.melt.html)__

# `melt()` hace lo opuesto a `pivot()`

# In[58]:


grades = pd.DataFrame([[6, 4, 5], [7, 8, 7], [6, 7, 9], [6, 5, 5], [5, 2, 7]], 
                       index = ['Mary', 'John', 'Ann', 'Pete', 'Laura'],
                       columns = ['test_1', 'test_2', 'test_3'])
grades.reset_index(inplace=True)
grades


# In[59]:


grades.melt(id_vars=['index']) # indicar las columna que identifican a cada entidad

