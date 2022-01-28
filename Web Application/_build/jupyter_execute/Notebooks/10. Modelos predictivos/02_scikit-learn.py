#!/usr/bin/env python
# coding: utf-8

# ## scikit-learn
# 
# ### Facilidad de uso
# 1. Crear el clasificador. Ex. DummyClassifier(strategy='most_frequent', random_state=0)
# 2. Entrenar el clasificador (fit)
# 3. Predecir con clasificador (predict)
# 
# ### Extensivo
# * Clasificación, regresión, clustering, reducción de dimensionalidad
# * Funciones para pre-procesamiento (reduccion de dimensionalidad, x ejemplo), 
# * Feature selection
# * evaluación del desempeño
# * Optimizado
# 
# ### Documentación
# __[scikit-learn](https://scikit-learn.org/stable/index.html)__

# ## Explorar datos de ejemplo
# #### Breast cancer dataset
# 
# Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import scipy as sp


# In[2]:


# Importar la funcion load_breast_cancer de sklearn.datasets 
from sklearn.datasets import load_breast_cancer


# In[3]:


breast_cancer_dataset = load_breast_cancer()


# In[4]:


print(breast_cancer_dataset.DESCR)


# In[5]:


#Diccionario de numpy arrays
breast_cancer_dataset.keys()


# In[6]:


breast_cancer_dataset.data


# In[7]:


breast_cancer_dataset.feature_names


# In[8]:


breast_cancer_dataset.data.shape


# In[9]:


breast_cancer_dataset.target


# In[10]:


breast_cancer_dataset.target_names


# In[11]:


breast_cancer_dataset.target.shape


# In[12]:


#Construir un DataFrame a partir de los componentes de breast_cancer_dataset
df_features = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
df_target = pd.DataFrame(breast_cancer_dataset.target, columns=["cancer"])


# In[13]:


df = pd.concat([df_features, df_target], axis=1)


# In[14]:


df.head()


# In[15]:


df.shape

