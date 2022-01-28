#!/usr/bin/env python
# coding: utf-8

# ## Ejercicio misceláneo: NLP, Web Scrapping (otra resolución)
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
import collections
from heapq import nlargest
import unicodedata

http = urllib3.PoolManager()


# In[2]:


articleURL = "https://www.muyinteresante.es/naturaleza/video/objetivo-2030-proteger-el-30-de-la-tierra-751579255112"


# In[3]:


import requests
respuesta = requests.get(articleURL) 
soup = BeautifulSoup(respuesta.text,"html.parser")


# In[4]:


text_containers = soup.body.find_all(name='div', class_='paragraph--text')


# In[5]:


#El valor de la variable text debe ser una cadena con los párrafos separados por '\n'
text = '';
for tc in text_containers:
    p_containers = tc.find_all(name='p')
    for p in p_containers:
        text = text + p.text + '\n'
text = text[26:]
text


# In[6]:


text = unicodedata.normalize("NFKD",text) #replace \xa0 with regular space
print(text)


# In[7]:


def summarize(text, n):
    sentences = sent_tokenize(text)
    assert n <= len(sentences)
    words = word_tokenize(text.lower())
    
    our_stopwords = set(stopwords.words('spanish') + list(punctuation) + ['“', '”'])
    
    words_without_stops=[word for word in words if word not in our_stopwords]
    
    #obtener un diccionario con la frecuencia de las palabras en words_without_stops
    word_frequencies = collections.Counter(words_without_stops)
    
    #Obtener los scores para cada oración de acuerdo a la reglas anteriormente descritas
    #sentence_scores es un diccionario con la posición/indice de cada párrafo y su score
    
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        peso = 0
        for word in word_tokenize(sentence.lower()):
            peso = peso + word_frequencies[word]
        sentence_scores[i] = peso

    sorted_sentence_scores = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
    sentences_with_score = [{'index': k, 'score': v} for k, v in sorted_sentence_scores]
    
    best_sentences = sentences_with_score[:n]
    indexes_of_the_highest_scores = [s['index'] for s in best_sentences]
    
    summary_sentences = []
    for index in indexes_of_the_highest_scores:
        summary_sentences.append(sentences[index])
    return summary_sentences


# In[8]:


summarize(text,3)

