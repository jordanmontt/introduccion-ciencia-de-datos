#!/usr/bin/env python
# coding: utf-8

# ## Arboles de decisión

# In[1]:


import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# In[8]:


df = pd.read_csv(os.path.join('../Datasets/diabetes.csv'))
df.head()


# In[11]:


feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = df[feature_cols]
Y = df["Outcome"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training, 30% test
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[13]:


# baseline no incluye poda (max_depth)
treev1 = DecisionTreeClassifier()
treev1.fit(X_train, Y_train)


# In[15]:


Y_pred = treev1.predict(X_test)
Y_pred


# In[28]:


def metricas_desempenio(tree):
    print('accuracy del clasificador - version 1 : {0:.2f}'.format(accuracy_score(Y_test, tree.predict(X_test))))
    print('matriz de confusión del clasificador - version 1: \n {0}'.format(confusion_matrix(Y_test, tree.predict(X_test))))
    print('precision del clasificador - version 1 : {0:.2f}'.format(precision_score(Y_test, tree.predict(X_test))))
    print('recall del clasificador - version 1 : {0:.2f}'.format(recall_score(Y_test, tree.predict(X_test))))
    print('f1 del clasificador - version 1 : {0:.2f}'.format(f1_score(Y_test, tree.predict(X_test))))
metricas_desempenio(treev1)


# In[29]:


#Ajustar algunos hiperparámetros
tree_v2 = DecisionTreeClassifier(criterion="entropy", max_depth=3)
tree_v2.fit(X_train, Y_train)


# In[30]:


metricas_desempenio(tree_v2)


# ### Hiperparámetros para ajustar la complejidad del modelo
# 
# __[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)__
# * class_weight=None importancia relativa de los valores de clasificación
# * criterion='entropy'/'gini'
# * max_depth=3 distancia max entre a raiz y las hojas
# * max_features=None numero max de variables a considerar
# * max_leaf_nodes=20 numero max de hojas
# * min_impurity_decrease=0.0
# * min_impurity_split=None (Deprecado)
# * min_samples_leaf=1 Podar si quedan menos que este numero de ejemplos 
# * min_samples_split=2 Continuar si quedan al menos esta cantidad de ejemplos
# * min_weight_fraction_leaf=0.0 Porcentaje minimo de ejemplo para continuar
# 
# 
# <img src="./img/28-complejidad-vs-accuracy.png" style="width:600px"/>
# 
# **Más allá de cierto umbral, la complejidad del modelo afecta negativamente el desempeño debido al sobreajuste**

# ## Sobreajuste
# 
# __[Sobreajuste](https://es.wikipedia.org/wiki/Sobreajuste)__
# 
# <img src="./img/29-overfitting-sobreajuste.png" style="width:600px"/>
# 
# <img src="./img/30.0-underfitting.png" style="width:600px"/>
# 
# <img src="./img/30.1-origen-del-sobreajuste.png" style="width:600px"/>
# 
# **Como evitar?**
# En el caso particular de los árboles de decisión, reducir nodos del arbol cuando no incrementan los indicadores con una buena cantidad de datos de prueba (**poda - pruning**)
# 
# ### Ensemble learning
# 
# __[Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning)__
# 
# <img src="./img/31-ensemble-learning.png" style="width:600px"/>
# 
# <img src="./img/32-ensemble-arboles.png" style="width:600px"/>
# 
# <img src="./img/33-ensemble-arboles.png" style="width:600px"/>
# 

# ## Random Forest
# 
# __[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)__
# 
# Se crean varios árboles **INDEPENDIENTES** variando los casos/observaciones del conjunto de entrenamiento y/o las variables empleadas durante el proceso de entrenamiento.
# 
# Las predicciones de cada modelo (árbol) tienen el mismo peso y el resultado final es el voto de mayoría
# 
# **Parámetros:**
# * **n_estimators** número de clasificadores, árboles en este caso.
# 
# Los valores adecuados para este y otros parámetros se obtienen via experimentación (prueba y error). Si es posible, se recomienda tener varios conjuntos de prueba para seleccionar el modelo con el mejor desempeño (promedio) en todos los conjunto de prueba

# In[31]:


from sklearn.ensemble import RandomForestClassifier
#Ajustar n_estimators puede reducir la posiblidad de overfitting
tree_v3 = RandomForestClassifier(n_estimators=10)
tree_v3


# In[34]:


tree_v3.fit(X_train, Y_train)


# In[35]:


metricas_desempenio(tree_v3)


# ## Gradient Boosted Trees
# Los arboles se construyen en **secuencia** a partir de una fracción del conjunto de entrenamiento; la idea central es que el siguiente árbol corriga los errores del anterior.:
# 
# Inicialmente, todos los ejemplos tienen la misma probabilidad de ser seleccionados. A partir del segundo árbol, los ejemplos que fueros incorrectamente clasificados por el árbol anterior tienen mayor probabilidad de ser seleccionados. (para detectar patrones que no fueron detectados por el anterior)
# 
# En consecuencia, cada árbol se crea a partir de una fracción diferente del conjunto de entrenamiento. En la colección final, la clasificación de cada árbol tiene un peso mayor en función del desempeño obtenido con el conjuto de entrenamiento.
# 
# __[py-xgboost](https://anaconda.org/anaconda/py-xgboost)__
# 
# __[xgboost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)__
# 

# In[38]:


from xgboost.sklearn import XGBClassifier


# In[39]:


tree_v4 = XGBClassifier(n_estimators=10)
tree_v4


# In[40]:


tree_v4.fit(X_train, Y_train)


# In[41]:


metricas_desempenio(tree_v4)


# ### Parámetros que se pueden emplear para evitar sobre ajuste (overfitting) 
# * **n_estimators**: a mayor cantidad de ejemplos, se puede incrementar el valor n_estimators para evitar sobreajuste
# 
# * **learning_rate**, determina la probabilidad de que un ejemplo sea seleccionado en la siguiente iteracion, se recomienda un valor entre 0.1 - 0.2 para reducir la probabilidad de que se produzca overfitting
# 
# * **subsample**, permite controlar el tamaño de la fracción del conjunto de entrenamiento  para cada iteración. Mientras más bajo el valor, más probabilidad hay de que los conjuntos de entrenamiento entre iteraciones sean diferentes (a mayor diferencia, menos probabilidad de que se produzca overfitting). Se recomienda valores entre 0.5 - 1.0
# 
# * **colsample_bytree**, permite controlar la fracción de las variables empleadas para entrenar los árboles en cada iteración. Se recomienda valores entre 0.5 - 1.0

# ## Optimización de parámetros
# 
# Objetivo: encontrar la mejor combinación de hiper-parámetros para obtener el clasificador con el mejor desempeño.
# 
# Para evitar probar manualmente todas las posibles combinaciones de valores para todos los posibles parámetros que resultan en un buen desempeño, se emplean técnicas de optimización para evitar buscar en todo el espacio de posible valores y garantizar al mismo tiempo un buen desempeño del clasificador. 
# 
# 
# __[hyperopt (Distributed Hyperparameter Optimization)](https://github.com/hyperopt/hyperopt)__ es el módulo python que facilita realizar esta tarea.

# In[42]:


#conda install -c conda-forge hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK,Trials


# In[43]:


space = {
    'x':hp.quniform('x',-1,1,1), #probar con valores entre -10 - 10, con incrementos de 1 
}


# In[44]:


def objective(params):
    x = int(params['x'])
    return {'loss':x ** 2,'status':STATUS_OK}   


# In[45]:


trials = Trials()


# In[46]:


best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=5)
print(best)


# In[51]:


#Probar valores entre 100 - 1000, con incrementos de 1 - con igual probabilidad de ser seleccionado: 
#'n_estimators':hp.quniform('n_estimators',100,1000,1)
#Crear un diccionario que contiene la configuración para generar diferentes valores para cada parámetro; en este ejemplo,
#para el algoritmo XGBClassifier.
space =  {
    'n_estimators':hp.quniform('n_estimators',100,1000,1), #probar con valores entre 100 - 100, con incrementos de 1 
    'learning_rate':hp.quniform('learning_rate',0.025,0.5,0.025),
    'max_depth':hp.quniform('max_depth',1,13,1),
    'subsample': hp.quniform('subsample',0.5,1,0.05),
    'colsample_bytree':hp.quniform('colsample_bytree',0.5,1,0.05),
    'nthread':6, #cuando se posible, paralelizar el procesamiento empleando hasta 6 hilos
    'silent':1 #si ocurre un error, continuar con la ejecución
}


# In[52]:


#Es necesario definir una función de manera tal que cuando alcance el valor mínimo, esto implique que el clasificador
#ha alanzado en mejor desempeño. 
#En el ejemplo siguiente, el menor valor posible para esta función (0), si se da cuando accuracy = 1.
def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])  
    clf = XGBClassifier(**params) #https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/
    clf.fit(X_train, Y_train)   
    accuracy = accuracy_score(Y_test, clf.predict(X_test))
    return {'loss': 1 - accuracy, 'status': STATUS_OK}


# In[53]:


#https://github.com/hyperopt/hyperopt/wiki/FMin#12-attaching-extra-information-via-the-trials-object
#fmin Itera 100 veces y retorna la combinación de parámetros que generan el menor valor para la función 'objective'
trials = Trials()
best = fmin(objective,space,algo=tpe.suggest,trials=trials,max_evals=100)
print(best)


# In[54]:


best['n_estimators'] = int(best['n_estimators'])
best['max_depth'] = int(best['max_depth'])                      


# In[55]:


tree_v5 = XGBClassifier(**best)
tree_v5


# In[56]:


tree_v5.fit(X_train, Y_train)


# In[57]:


metricas_desempenio(tree_v5)

