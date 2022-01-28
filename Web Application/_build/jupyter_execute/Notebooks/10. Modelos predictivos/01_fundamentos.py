#!/usr/bin/env python
# coding: utf-8

# ## Aprendizaje Automático (Machine Learning)
# 
# ### ¿Qué es aprender?
# 
# __[Aprendizaje Automático](https://es.wikipedia.org/wiki/Aprendizaje_autom%C3%A1tico)__
# 
# 
# ### Ejemplo de un problema de clasificación supervisado
# 
# <img src="./img/01_spam-detection.png" style="width:600px"/>
# 
# ### Variables independientes (features) y variable objetivo (response/tag/class)
# 
# <img src="./img/02_features-input-output.png" style="width:600px"/>
# 
# <img src="./img/03_feature.png" style="width:600px"/>
# <hr\>
# 
# El proceso de "entrenamiento" busca obtener un artefacto que sea capaz de clasificar nuevas observaciones/casos. A mayor cantidad de ejemplos, se espera un mejor desempeño.
# 
# <img src="./img/04_generalizacion.png" style="width:600px"/>
# 
# ### Aprendizaje supervisado (clasificación vs regresión)
# 
# __[Algoritmos de aprendizaje supervisado](https://en.wikipedia.org/wiki/Supervised_learning#Algorithms)__
# 
# <img src="./img/05_supervised_classification.png" style="width:600px"/>
# <hr\>
# <img src="./img/06_supervised_regression.png" style="width:600px"/>
# 
# ### Aprendizaje no supervisado
# 
# <img src="./img/07_unsupervised-clustering.png" style="width:600px"/>
# 

# ## Aprendizaje Automático vs Sistemas Basados en Reglas
# 
# <img src="./img/03_ml_vs_rules.png" style="width:700px"/>
# 
# ML. Pro: adecuado (probablemente la única alternativa) en escenarios con reglas dinámicas y dependiente del contexto. Con: requiere cantidad considerable de datos históricos de buena calidad y la selección de buenos vectores de características (esto requiere experiencia y conocimiento de el funcionamiento de los algoritmos de ML)
# 
# Reglas. Pro: adecuado en escenarios con reglas relativamente estáticas o cuando no se dispone de cantidad considerable de datos históricos (en cuyo caso es la única opción). Con: pobre desempeño a la hora de capturar reglas que dependen del contexto, es muchos escencario se requiere de expertos que provean las reglas.

# ## Evaluación del desempeño de un clasificador (binario)
# 
# ### Accuracy (Exactitud)
# 
# <img src="./img/08_accuracy.png" style="width:700px"/>
# 
# __[Cuando la exactitud no es suficiente](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)__
# 
# __[Matriz de Confusión](https://es.wikipedia.org/wiki/Matriz_de_confusi%C3%B3n)__
# 
# 
# ###  Precisión
# <img src="./img/09_precision.png" style="width:700px"/>
# 
# **¿Qué proporción de clasificaciones positivas fue correcta?**
# 
# ###  Recall (Exhaustividad) 
# <img src="./img/10_recall.png" style="width:600px"/>
# 
# 
# **¿Qué proporción de positivos reales se clasificó correctamente?**
# 
# ###  Preguntas sobre precisión y exhaustividad
# 
# Con un clasificador (digamos un test para detectar una enfermedad) de alta exhaustividad, es más o menos probable fallar en la detección de la enfermedad?
# 
# Con un clasificador (digamos un test para detectar una enfermedad) de alta exhaustividad, es más o menos probable detectar la enfermedad en personas que no la tienen?
# 
# Si la enfermedad es mortal, quá es preferible: alta Presición o alta Exhaustividad?

# ## Modelo - Linea base

# In[1]:


import pandas as pd
import numpy as np
import os


# In[30]:


df_entrenamiento = pd.read_csv(os.path.join("csv", "train.csv"), index_col='PassengerId')


# In[27]:


df_entrenamiento.info()


# Hacemos limpieza de las columnas que no son necesarias para este ejercicio.

# In[31]:


df_entrenamiento = df_entrenamiento.drop(['Ticket', 'Embarked', 'Cabin'], axis=1)


# In[32]:


df_entrenamiento.head()


# ## Division Entrenamiento / Test (train - test split)

# Separar las caracteristicas de la variable de salida
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html

# In[33]:


X = df_entrenamiento.loc[:,'Age':].to_numpy().astype('float') #convertir a una matriz numpy
y = df_entrenamiento['Survived'].ravel() #convertir a un vector numpy


# In[34]:


print(X.shape, y.shape)


# In[35]:


print(type(X), type(y))


# In[37]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Ver si no se trata de un problema con clases muy "desbalanceadas"

# In[39]:


print('media de supervivencia en el conjunto de entrenamiento : {0:.3f}'.format(np.mean(y_train)))
print('media de supervivencia en el conjunto de prueba : {0:.3f}'.format(np.mean(y_test)))


# ## Clasificador linea base

# importar el algoritmo de clasificación

# In[40]:


from sklearn.dummy import DummyClassifier


# Crear el clasificador

# In[41]:


clasificador_lineabase = DummyClassifier(strategy='most_frequent', random_state=0)


# "entrenar" el clasificador

# In[42]:


clasificador_lineabase.fit(X_train, y_train)


# El método `score` devuelve Accuracy (exactitud) del clasificador linea base
# Accuracy obtenido sólo con devolver la clasificación con mayor frecuencia
# Se debe superar este valor empleando Machine Learning.

# In[43]:


print('accuracy del clasificador : {0:.2f}'.format(clasificador_lineabase.score(X_test, y_test)))


# In[44]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# La función compara las clasificaciones conocidas (del conjunto de prueba) 
# con las predicciones hechas por el clasificador para los ejemplos del conjunto de prueba

# In[45]:


print('accuracy del clasificador : {0:.2f}'.format(accuracy_score(y_test, clasificador_lineabase.predict(X_test))))


# Matriz de confusión

# In[46]:


print('matriz de confusión del clasificador: \n {0}'.format(confusion_matrix(y_test, clasificador_lineabase.predict(X_test))))


# In[47]:


# precision y recall
print('precision del clasificador : {0:.2f}'.format(precision_score(y_test, clasificador_lineabase.predict(X_test))))
print('recall del clasificador : {0:.2f}'.format(recall_score(y_test, clasificador_lineabase.predict(X_test))))

