#!/usr/bin/env python
# coding: utf-8

# ## Regresión logística
# 
# Es un algoritmo para obtener un clasificador binario. 
# 
# La regresión logística es bastante efectiva en situaciones en las que la relación entre la **probabilidad** de lograr una meta/objetivo (Y) está vinculada a los recursos necesarios (X) de manera no lineal donde una disminución/aumento de cierto recurso más allá de cierto umbral disminuye/aumenta drásticamente la probabilidad de lograr el objetivo.
# 
# 
# <img src="./img/01-lineal-vs-logistica.png" style="width:600px"/>
# 
# 
# <img src="./img/02-regresion-logistica.png" style="width:600px"/>
# 
# 
# Los clasificadores binaros basados en regresión logística clasifican las observaciones de acuerdo a un umbral típicamente 0.5 (50%).
# 
# Hay dos técnicas comunmente empleadas para obtener los coeficientes de regresión. __[MLE](https://es.wikipedia.org/wiki/M%C3%A1xima_verosimilitud)__ y __[mínimos cuadrados](https://es.wikipedia.org/wiki/M%C3%ADnimos_cuadrados)__ (luego de convertir la relación establecida por la curva "S" a una relación lineal)
# 
# 
# __[Scikit Learn - Regresión logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)__

# In[1]:


import pandas as pd
import numpy as np
import os


# In[21]:


df_entrenamiento = pd.read_csv(os.path.join("csv", "train.csv"), index_col='PassengerId')


# In[14]:


df_entrenamiento.head()


# Hacemos limpieza de las columnas que no son necesarias para este ejercicio.

# In[22]:


df_entrenamiento = df_entrenamiento.drop(['Ticket', 'Embarked', 'Cabin'], axis=1)
df_entrenamiento = df_entrenamiento.dropna()


# In[23]:


X = df_entrenamiento.loc[:,'Age':].to_numpy().astype('float')
y = df_entrenamiento['Survived'].ravel() 


# In[24]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


# crear el clasificador
clasificador_reg_log = LogisticRegression(random_state=0, solver='liblinear')


# In[25]:


# entrenar el clasificador
clasificador_reg_log.fit(X_train,y_train)


# In[26]:


print('accuracy del clasificador - version 1 : {0:.2f}'.format(clasificador_reg_log.score(X_test, y_test)))


# ### El hiperparámetro 'penalty'
# __[L1 Norms versus L2 Norms](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms)__
# 
# __[L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)__

# In[27]:


#evaluar el desempeño
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# In[28]:


# accuracy
print('accuracy del clasificador - version 1 : {0:.2f}'.format(accuracy_score(y_test, clasificador_reg_log.predict(X_test))))
# confusion matrix
print('matriz de confusión del clasificador - version 1: \n {0}'.format(confusion_matrix(y_test, clasificador_reg_log.predict(X_test))))
# precision 
print('precision del clasificador - version 1 : {0:.2f}'.format(precision_score(y_test, clasificador_reg_log.predict(X_test))))
# recall 
print('recall del clasificador - version 1 : {0:.2f}'.format(recall_score(y_test, clasificador_reg_log.predict(X_test))))
# f1
print('f1 del clasificador - version 1 : {0:.2f}'.format(f1_score(y_test, clasificador_reg_log.predict(X_test))))


# In[29]:


# coeficientes del modelo
clasificador_reg_log.coef_


# In[30]:


df_entrenamiento.loc[:,'Age':].columns


# In[31]:


list(zip(df_entrenamiento.loc[:,'Age':].columns, clasificador_reg_log.coef_[0]))

