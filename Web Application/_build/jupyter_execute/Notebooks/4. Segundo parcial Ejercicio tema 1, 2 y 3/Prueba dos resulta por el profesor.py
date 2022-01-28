#!/usr/bin/env python
# coding: utf-8

# ## Ejercicio misceláneo: NLP, Web Scrapping 
# 
# La prueba consiste en completar el código marcado con **#FIXME** para obtener automáticamente el resumen de un texto.
# 
# La decripción general de las reglas/algoritmo:
# 
# - Identificar las frecuencias de las palabras en el texto.
# - Obtener las oraciones/párrafos en el texto.
# - Obtener un score para cada oración/párrafo que indique su importancia en el texto, tendrán mayor score aquellas oraciones que incluyan las palabras con mayor frecuencia.
# - Seleccionar las 3 oraciones con el score más alto para construir el resumen.

# In[1]:


import urllib3
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from collections import defaultdict
from heapq import nlargest
import unicodedata

http = urllib3.PoolManager()


# In[2]:


articleURL = "https://www.muyinteresante.es/naturaleza/video/objetivo-2030-proteger-el-30-de-la-tierra-751579255112"


# In[3]:


import requests
respuesta = requests.get(articleURL) 
soup = BeautifulSoup(respuesta.text,"html.parser")
soup


# In[4]:


p_containers = soup.body.find_all(name="div", attrs={"class":"paragraph--text"})


# In[5]:


p_containers[0].find_all("p")[1]


# In[6]:


#El valor de la variable text debe ser una cadena con los párrafos separados por '\n'
text = '';
for p_container in p_containers:
    text = text + '\n'.join(map(lambda p: p.text, p_container.find_all("p"))) #function


# In[7]:


text = unicodedata.normalize("NFKD",text) #replace \xa0 with regular space
text


# In[8]:


def summarize(text, n):
    sentences = sent_tokenize(text)
    
    assert n <= len(sentences)
    words = word_tokenize(text.lower())
    our_stopwords = set(stopwords.words('spanish') + list(punctuation) + ['“', '”'])
    
    words_without_stops=[word for word in words if word not in our_stopwords]
    
    #obtener un diccionario con la frecuencia de las palabras en words_without_stops
    word_frequencies = FreqDist(words_without_stops)  # collections.Counter
       
    #Obtener los scores para cada oración de acuerdo a la reglas anteriormente descritas
    #sentence_scores es un diccionario con la posición/indice de cada párrafo y su score
    sentence_scores = defaultdict(int) 
    for i,sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            #sentence_scores[i] += word_frequencies.get(word,0)
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]
             
        
    indexes_with_high_score =  nlargest(n, sentence_scores, key=sentence_scores.get) #function
    summary_sentences =[sentences[j] for j in sorted(indexes_with_high_score)] #function
    return summary_sentences


# In[9]:


summarize(text,3)

