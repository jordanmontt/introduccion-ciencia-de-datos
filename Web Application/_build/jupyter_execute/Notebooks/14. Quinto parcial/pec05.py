#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Habilitar intellisense
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')


# ## Arboles de decisión

# In[2]:


import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pickle


# In[21]:


df = pd.read_csv(os.path.join("diabetes.csv"))
df.head()


# In[22]:


feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = df[feature_cols] # Features
y = df["Outcome"] # Target variable


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training, 30% test


# In[24]:


#Ajustar n_estimators puede reducir la posiblidad de overfitting
rftree = RandomForestClassifier(n_estimators=10)
rftree


# In[25]:


rftree.fit(X_train,y_train)


# In[26]:


# métricas de desempeño
# accuracy
print('accuracy del clasificador - version 3 : {0:.2f}'.format(accuracy_score(y_test, rftree.predict(X_test))))
# confusion matrix
print('matriz de confusión del clasificador - version 3: \n {0}'.format(confusion_matrix(y_test, rftree.predict(X_test))))
# precision 
print('precision del clasificador - version 3 : {0:.2f}'.format(precision_score(y_test, rftree.predict(X_test))))
# precision 
print('recall del clasificador - version 3 : {0:.2f}'.format(recall_score(y_test, rftree.predict(X_test))))
# f1
print('f1 del clasificador - version 3 : {0:.2f}'.format(f1_score(y_test, rftree.predict(X_test))))


# In[27]:


ruta_archivo_clasificador = os.path.join("classifier.pkl")


# In[28]:


archivo_clasificador = open(ruta_archivo_clasificador, "wb")


# In[29]:


pickle.dump(rftree, archivo_clasificador)


# In[30]:


archivo_clasificador.close()

