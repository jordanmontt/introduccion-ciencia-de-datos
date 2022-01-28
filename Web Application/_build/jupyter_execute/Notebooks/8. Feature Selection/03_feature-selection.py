#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Feature selection (selección de variables)
# 
# __[More data beats clever algorithms, but better data beats more data](https://quotefancy.com/quote/1779267/Peter-Norvig-More-data-beats-clever-algorithms-but-better-data-beats-more-data)__
# 
# 
# ## Definición
# __[Selección de variable](https://es.wikipedia.org/wiki/Selecci%C3%B3n_de_variable)__
# 
# Proceso manual o automático de identificación de las variables que contribuyen o explican en mayor grado los cambios en la variable dependiente (variable de predicción). Incluir variables irrelevantes afecta negativamente el desempeño de los modelos preditivos porque les inducen a detectar "patrones" basados en aspectos irrelevantes (mayor probabilidad de sobreajuste en los modelos).
# 
# Además, eliminar variables innecesarias o redundantes implica otras ventajas:
# 
# - Simplificación de la descripción de cada observación/ejemplo. 
# Modelos complejos se sobreajustan con mayor facilidad.
# - Tiempo de entrenamiento menor (hay menos datos que procesar)
# 
# ## Enfoques
# 
# ### Filtrado a priori (filtering)
# 
# Estas técnicas buscan eliminar las variables innecesarias **antes** de emplear los ejemplos/observaciones para entrenar el modelo predictivo. 
# 
# Estás técnicas son computacionalmente menos costosas que las técnicas wrapper pero no toman en cuenta las caracterísicas del modelo y, por lo tanto, existe la posiblidad de excluir variables que puedan ser útiles. Por lo tanto, se recomienda emplearlas de manera conservadora o evaluar los resultados obtenidos con diferentes conjuntos de variables contra una __[linea base](https://en.wikipedia.org/wiki/Kitchen_sink_regression)__ en la que se emplean todas las variables. Tampoco identifican situaciones de **multicolinealidad** que pueden tener efectos negativos en modelos de regresión.
# 
# Referencias:
# 
# - __[Multicollinearity: Why is it a problem?](https://towardsdatascience.com/multicollinearity-why-is-it-a-problem-398b010b77ac)__
# - __[Multicollinearity in Regression Analysis: Problems, Detection, and Solutions](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)__
# 
# 
# <img src="./img/03-feature-selection-filtering.png" style="width:600px"/>
# 
# 
# ### Wrapper
# 
# Estas técnicas buscan eliminar las variables innecesarias **después** de emplear los ejemplos/observaciones para entrenar el modelo predictivo y en función de su efecto en el desempeño del mismo. La idea es quedarnos con el subconjunto de las variables que maximizen el desempeño del modelo.
# 
# <img src="./img/04-feature-selection-wrapper.png" style="width:600px"/>
# 
# ### Incrustados (Embedded)
# 
# Se realiza la selección de las variables como parte del proceso de entrenamiento del modelo. Por ejemplo, en los modelos de clasificación basados en árboles de decisión, el algoritmo que construye el árbol incopora la selección de las variables que  clasifican la mayor cantidad de ejemplos.
# 
# ## Técnicas de filtrado a priori
# 
# - Umbral de Varianza (Variance Threshold), para eliminar variables con varianza inferior a un umbral. Mientras más cercana a cero la varianza de una variable, ésta suele ser menos útil para predecir el valor de Y.
# - __[Prueba chi cuadrado](https://es.wikipedia.org/wiki/Prueba_%CF%87%C2%B2_de_Pearson)__ Para cada variable X, emplear la prueba para determinar si dicha variable y Y son independientes. Si lo son, eliminar X. Si por el contrario, cambios en la variable X provocan cambios significativos en el valor esperado de la variable Y, X se considera signinifiva. Se aplica cuando Y y X son variables categóricas (clasificación)
# - __[Información mutua](https://es.wikipedia.org/wiki/Informaci%C3%B3n_mutua)__ determina la dependencia mutua de dos variables midiendo la reducción de la incertidumbre (entropía) de una variable aleatoria X, debido al conocimiento del valor de otra variable aleatoria Y. Un valor elevado es una indicación de que una variable tiene gran influencia en la otra, cero indica que son  independientes. Se considera una técnica superior a ANOVA (Analysis of Variance) por su capacidad capturar relaciones no lineales.

# ![](./01-eda-visual-techniques.png)

# In[18]:


import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


# ## Umbral de Varianza
# Se puede trabajar también con datos normalizados (z-score)

# In[19]:


df = pd.read_csv(os.path.join("boston.csv"))
df.shape


# In[3]:


df.head()


# In[20]:


X = df.drop('Median Home Value', axis = 1)
y = df['Median Home Value']
X.head()


# In[21]:


X.var(axis = 0)


# In[13]:


X.shape


# In[23]:


#No es muy complicado crear un función equivalente
from sklearn.feature_selection import VarianceThreshold

select_features = VarianceThreshold(threshold = 8.0)


# In[24]:


X_new = select_features.fit_transform(X)
X_new


# In[25]:


X_new.shape


# In[26]:


X_new = pd.DataFrame(X_new)
X_new.head()


# In[27]:


selected_features = []

for i in range(len(X_new.columns)):
    for j in range(len(X.columns)):

        if(X_new.iloc[:,i].equals( X.iloc[:,j])):
            selected_features.append(X.columns[j])
            
selected_features


# In[28]:


rejected_features = set(list(X)) - set(selected_features)
rejected_features


# In[13]:


X.var(axis = 0)


# In[29]:


X_new.columns = selected_features


# In[30]:


X_new.head()


# In[27]:


X.head()


# ## Prueba Chi cuadrado

# In[42]:


df = pd.read_csv(os.path.join("diabetes.csv"))
df.head()


# In[43]:


df.shape


# In[44]:


X = df.drop('Outcome', axis = 1)
y = df['Outcome']


# In[45]:


X.shape


# In[46]:


X.dtypes


# In[47]:


X = X.astype(np.float64)


# In[48]:


#chi2 determina si cada X,y son indepedientes; 'y' debe ser una variable categórica.
#SelectKBest emplea chi2 para probar cada (X,y) y seleccionar las K mejore variables
from sklearn.feature_selection import chi2, SelectKBest

select_features = SelectKBest(chi2, k=3)

X_new = select_features.fit_transform(X, y)


# In[49]:


X_new.shape


# In[50]:


X_new = pd.DataFrame(X_new)

X_new.head()


# In[38]:


X_new.dtypes


# In[51]:


selected_features = []

for i in range(len(X_new.columns)):
    for j in range(len(X.columns)):
        
        if(X_new.iloc[:,i].equals(X.iloc[:,j])): #requiere de X = X.astype(np.float64)
            selected_features.append(X.columns[j])
            
selected_features


# In[52]:


rejected_features = set(list(X)) - set(selected_features)

rejected_features


# In[53]:


X_new.columns = selected_features
X_new.head()


# In[54]:


X.head()


# ## Información mutua

# In[55]:


df = pd.read_csv(os.path.join("diabetes.csv"))
df.head()


# In[56]:


X = df.drop('Outcome', axis = 1)
y = df['Outcome']


# In[58]:


X = X.astype(np.float64)


# `mutual_info_classif` determina si cada `X,y` son indepedientes. 
# 
# Si 'y' es una variable numérica contínua se debe usar mutual_info_regression.
# SelectPercentile emplea `mutual_info_classif` para probar cada `(X,y)` y seleccionar las que estén por encima
# un valor de 100 retiene todas la variables, un valor de 10 retiene aproximadamente la décima parte de las variables

# In[66]:


from sklearn.feature_selection import mutual_info_classif, SelectPercentile

select_features = SelectPercentile(mutual_info_classif, percentile = 30)

X_new = select_features.fit_transform(X, y)


# In[67]:


X_new.shape


# In[68]:


X_new = pd.DataFrame(X_new)

X_new.head()


# In[70]:


selected_features = []

for i in range(len(X_new.columns)):
    for j in range(len(X.columns)):
        
        if(X_new.iloc[:,i].equals(X.iloc[:,j])): #requiere de X = X.astype(np.float64)
            selected_features.append(X.columns[j])
            
selected_features


# In[71]:


rejected_features = set(list(X)) - set(selected_features)

rejected_features


# In[72]:


X_new.columns = selected_features
X_new.head()


# In[73]:


X.head()

