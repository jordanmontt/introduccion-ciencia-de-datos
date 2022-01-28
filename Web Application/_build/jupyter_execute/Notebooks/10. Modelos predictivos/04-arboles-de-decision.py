#!/usr/bin/env python
# coding: utf-8

# ## Arboles de decisión

# Un árbol de decisión es un clasificador que toma como entrada una entidad descrita por un conjunto de atributos y devuelve una «decisión». Los atributos de entrada pueden ser discretos o continuos. 
# 
# Un árbol de decisión aplica una secuencia de tests para poder alcanzar la decisión. Cada nodo interno del árbol corresponde con un test sobre el valor de una de las propiedades, y las ramas que salen del nodo están etiquetadas con los posibles valores de dicha propiedad. Cada nodo hoja del árbol representa el valor que ha de ser devuelto si dicho nodo hoja es alcanzado. 
# 
# La representación en forma de árboles de decisión es muy natural para los humanos; muchos manuales que explican cómo hacer determinadas tareas (por ejemplo, reparar un coche) están escritos en su totalidad como un árbol de decisión.
# 
# 
# <img src="./img/21-arbol-decision.png" style="width:600px"/>

# ## Aprendizaje de Arboles de decisión
# __[Aprendizaje basado en árboles de decisión](https://es.wikipedia.org/wiki/Aprendizaje_basado_en_%C3%A1rboles_de_decisi%C3%B3n)__
# 
# <img src="./img/22-ejemplos-arbol-decision.png" style="width:600px"/>
# 
# La idea básica del algoritmo APRENDIZAJE-ÁRBOL-DECISIÓN es realizar primero el test sobre el atributo más importante. Se considera como «atributo más importante» aquel que clasifica la mayor cantidad de ejemplos. De esta forma, esperamos obtener la clasificación correcta con un número reducido de tests; es decir, que todos los caminos en el árbol sean cortos y así el árbol completo será pequeño.
# 
# <img src="./img/23-atributo-importante.png" style="width:600px"/>
# 
# <img src="./img/24-algoritmo-aprendizaje-arbol.png" style="width:600px"/>
# 
# <img src="./img/25-contenido-informacion-atributo.png" style="width:600px"/>

# In[1]:


import math

def cantidad_info(*ps):   
    return sum([-p * (0 if p == 0 else math.log2(p)) for p in ps])


# No sabemos nada sobre la moneda, necesitamos 1 bit de información para responder la pregunta sobre si saldrá cara o cruz

# In[2]:


cantidad_info(1/2, 1/2)


# Sabemos que la moneda está cargada para que salga cara 9 de cada 10 veces.
# Necesitamos 1 bit de información para responder la pregunta sobre si saldrá cara o cruz

# In[3]:


cantidad_info(9/10, 1/10)


# Ssabemos que la moneda está cargada para que salga siempre cara, 

# In[4]:


cantidad_info(1, 0)


# ## Ganancia de información
# 
# <img src="./img/26-cantidad-informacion-inicial.png" style="width:600px"/>
# 
# 
# <img src="./img/27-ganacia-informacion.png" style="width:600px"/>

# In[5]:


# si = 6/12, no = 6/12
cantidad_de_info_inicial = cantidad_info(6/12, 6/12)
cantidad_de_info_inicial


# <img src="./img/22-ejemplos-arbol-decision.png" style="width:600px"/>
# 
# <img src="./img/23-atributo-importante.png" style="width:600px"/>

# In[6]:


#francés 2 casos;italiano 2 casos;tai 4 casos;hamburguesería 4 casos
info_tipo = 2/12*cantidad_info(1/2, 1/2) + 2/12*cantidad_info(1/2, 1/2) + 4/12*cantidad_info(2/4,2/4)  + 4/12*cantidad_info(2/4,2/4)
info_tipo


# In[7]:


#Ganacia tipo
cantidad_de_info_inicial - info_tipo


# In[8]:


#ninguno 2 casos;algunos 4 casos;lleno 6 casos
info_clientes = 2/12*cantidad_info(0, 1) + 4/12*cantidad_info(1, 0) + 6/12*cantidad_info(2/6,4/6)
info_clientes


# In[9]:


#Ganacia número de clientes
cantidad_de_info_inicial - info_clientes


# In[10]:


#Calcule la ganacia de información para la variable 'hambriento'
info_hambriento =7/12*cantidad_info(5/7, 2/7) + 5/12*cantidad_info(1/5, 4/5)
ganancia_hambriento = cantidad_de_info_inicial - info_hambriento
ganancia_hambriento


# <img src="./img/22.c-ejercicio-arbol-decision.png" style="width:600px"/>
