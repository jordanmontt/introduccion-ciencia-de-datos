#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Habilitar intellisense
get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')


# ## Ejemplo

# In[2]:


import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pickle
import requests, json


# In[3]:


ruta_archivo = os.path.join("diabetes.csv")
print(ruta_archivo)
df = pd.read_csv(ruta_archivo)


# In[4]:


feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_cols] # Features
y = df.Outcome # Target variable


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training, 30% test


# In[6]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ## Random Forest

# In[7]:


tree_v3 = RandomForestClassifier(n_estimators=10,n_jobs=2)


# In[8]:


tree_v3.fit(X_train,y_train)


# In[10]:


ruta_archivo_clasificador = os.path.join("clasificador", "Classifier.pkl")

#Abrir el archivo para escribir contenido binario
archivo_clasificador = open(ruta_archivo_clasificador, "wb")

pickle.dump(tree_v3, archivo_clasificador,protocol=2)

archivo_clasificador.close()


# In[11]:


url = 'http://127.0.0.1:8000/predict'
data = json.dumps({'Pregnancies':6,'Glucose':148,'BloodPressure':72,
                   'SkinThickness':35,'Insulin':0,'BMI':33.6,'DiabetesPedigreeFunction':0.627,'Age':50})
r = requests.post(url, data)

print(r)
print(r.text)


# In[11]:


df


# In[ ]:




