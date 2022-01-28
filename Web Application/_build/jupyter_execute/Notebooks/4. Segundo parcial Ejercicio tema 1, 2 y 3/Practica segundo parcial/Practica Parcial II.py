#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Obtener las 10 palabras que más aperecen en el siguiente artículo (ordenadas de manera descendente por su frecuencia). 
# Antes de obtener las frecuencias de las palabras deberá normalizar el texto aplicando las siguientes transformaciones:
# eliminar puntuación y palabras de parada
# convertir todo el texto a minísculas
#https://es.wikipedia.org/wiki/Pandemia_de_enfermedad_por_coronavirus_de_2020_en_Bolivia


# In[2]:


import urllib3
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from collections import defaultdict
import collections
from heapq import nlargest
import re
import requests
import nltk
import string

http = urllib3.PoolManager()


# In[203]:


articleURL = "https://es.wikipedia.org/wiki/Pandemia_de_enfermedad_por_coronavirus_de_2020_en_Bolivia"


# ## Web Scraping
# Obtener el texto de la pagina de Wikipedia

# In[247]:


def obtener_contenedor_texto(pagina):
    return pagina.find(name='div', attrs={'class': 'mw-parser-output'})

def quitar_tablas(contenedor_texto):
    for c in contenedor_texto.find_all('table'): 
        c.decompose()
    return contenedor_texto

def quitar_indice(contenedor_texto):
    contenedor_texto.find('div', attrs={'class', 'toc'}).decompose()
    return contenedor_texto

def quitar_titulos(contenedor_texto):
    for c in contenedor_texto.find_all('h2'):
        c.decompose()    
    for c in contenedor_texto.find_all('h3'):
        c.decompose()
    for c in contenedor_texto.find_all('h1'):
        c.decompose()
    return contenedor_texto

def quitar_corchete(contenedor_texto):
    for c in contenedor_texto.find_all('sup', {'class':'reference separada'}): 
        c.decompose()
    return contenedor_texto

def quitar_referencias(contenedor_texto):
    for c in contenedor_texto.find_all('div', {'class':'listaref'}): 
        c.decompose()
    return contenedor_texto

def quitar_styles(contenedor_texto):
    for c in contenedor_texto.find_all('style'): 
        c.decompose()
    return contenedor_texto

def quitar_elementos_intempestivos(contenedor_texto):
    contenedor_texto = quitar_tablas(contenedor_texto)
    contenedor_texto = quitar_indice(contenedor_texto)
    contenedor_texto = quitar_titulos(contenedor_texto)
    contenedor_texto = quitar_referencias(contenedor_texto)
    contenedor_texto = quitar_corchete(contenedor_texto)
    contenedor_texto = quitar_styles(contenedor_texto)  
    return contenedor_texto


# In[248]:


respuesta = requests.get(articleURL)
pagina = BeautifulSoup(respuesta.text,"html.parser")

contenedor_texto = obtener_contenedor_texto(pagina)
documento_tempestivo = quitar_elementos_intempestivos(contenedor_texto)
texto = documento_tempestivo.text
texto = texto.replace('\u200b', ' ').replace('\n', ' ').replace('\xa0000', ' ').replace('\xa0', ' ').replace('↑','')


# ## Convertir el texto a minúsculas

# In[244]:


texto = texto.lower()


# ## Eliminar las palabras de parada junto con los signos de puntuación

# In[252]:


# Obtener las palabras de parada y los signos de puntuacion en una sola lista
palabras_parada = nltk.corpus.stopwords.words('spanish')
signos_puntuacion = list(string.punctuation)
palabras_intempestivas = set( palabras_parada + signos_puntuacion)
# Tokenizar el texto
texto_tokenisado = nltk.word_tokenize(texto)
# Eliminamos las palabras de parada del texto
texto_sin_palabras_de_parada = [p for p in texto_tokenisado if p not in palabras_intempestivas]


# ## Obtener una lista ordenada ascendentemente con las 10 palabras que más aparecen el en texto

# In[260]:


# Tekenizar el texto según las palabras
count_by_token = collections.Counter(texto_sin_palabras_de_parada)
# Obtener la lista ordenada de frecuencias
frequency = [k for k, v in sorted(count_by_token.items(), key=lambda item: item[1], reverse=True)]
# Filtrar los últimos 10 resultados
frequency[0:10]

