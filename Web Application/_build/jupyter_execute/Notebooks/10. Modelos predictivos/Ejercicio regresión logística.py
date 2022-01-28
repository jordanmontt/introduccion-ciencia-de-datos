#!/usr/bin/env python
# coding: utf-8

# ## Ejercicio Regresión Logística
# 
# Crear un clasificador basado en el algoritmo de regresión logistica para predecir si el valor de la vivienda supera la media
# 
# entrada: housing.csv
# 
# Procedimiento:
# - Cargar los datos los datos a un DataFrame y explorar brevemente
# - Eliminar las observaciones que tengan algun dato faltante
# - Eliminar las observaciones con el valor atípico (max) para la variable 'median_house_value'
# - Aplicar one hot encoding a la variable 'ocean_proximity'
# - Crear una nueva variable boolean 'above_median'
# - Aplicar los pasos train-test-split para poder entrenar y evaluar el clasificador
# 
# Cuáles los valores para accuracy, matriz de confusion, precision, recall y f1 del clasificador?

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston


# In[2]:


# Cargar los datos los datos a un DataFrame y explorar brevemente
housing_csv_df = pd.read_csv(os.path.join("", "housing.csv"))
housing_csv_df.head()


# In[3]:


# Eliminar las observaciones que tengan algun dato faltante
housing_csv_df.loc[housing_csv_df['total_bedrooms'].isnull()]
housing_sin_valores_na = housing_csv_df.dropna(how='any')


# In[4]:


# Eliminar las observaciones con el valor atípico (max) para la variable 'median_house_value'
housing_sin_valores_na['expected_house_value'].plot.box()


# In[5]:


q1 = housing_sin_valores_na['expected_house_value'].quantile(.25)
q3 = housing_sin_valores_na['expected_house_value'].quantile(.75)
iqr = q3 - q1
indices_val_atipicos_max = housing_sin_valores_na.loc[housing_sin_valores_na['expected_house_value'] > q3 + 1.5*iqr].index
housing_sin_val_atip = housing_sin_valores_na.drop(indices_val_atipicos_max, axis=0)
housing_sin_val_atip['expected_house_value'].plot.box()


# In[6]:


# Aplicar one hot encoding a la variable 'ocean_proximity'
housing_procesado = pd.get_dummies(housing_sin_val_atip, columns=['ocean_proximity'])
housing_procesado


# In[7]:


# Crear una nueva variable boolean 'above_median'
media_expexted_house_value = housing_procesado['expected_house_value'].mean()
housing_procesado['above_median'] = housing_procesado['expected_house_value'] > media_expexted_house_value
housing_procesado


# In[8]:


# Aplicar los pasos train-test-split para poder entrenar y evaluar el clasificador
X = housing_procesado.drop(columns=['expected_house_value']).to_numpy().astype('float')
y = housing_procesado['expected_house_value'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


# Se crea al clasificador
clasificador_reg_log = LogisticRegression(random_state=0, solver='liblinear')
# Se entrena al clasificador
clasificador_reg_log.fit(X_train, y_train)


# In[ ]:


# accuracy
print('accuracy del clasificador: {0:.2f}'.format(accuracy_score(y_test, clasificador_reg_log.predict(X_test))))
# confusion matrix
print('matriz de confusión del clasificador: \n {0}'.format(confusion_matrix(y_test, clasificador_reg_log.predict(X_test))))
# precision 
print('precision del clasificador: {0:.2f}'.format(precision_score(y_test, clasificador_reg_log.predict(X_test))))
# recall 
print('recall del clasificador: {0:.2f}'.format(recall_score(y_test, clasificador_reg_log.predict(X_test))))
# f1
print('f1 del clasificador: {0:.2f}'.format(f1_score(y_test, clasificador_reg_log.predict(X_test))))


# In[ ]:




