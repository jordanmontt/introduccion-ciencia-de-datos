#!/usr/bin/env python
# coding: utf-8

# # Utilizando web scraping en una página real

# Este ejercicio consiste en extraer la información más importante de una página web. Para este ejercicio elegimos un artículo sobre el transhumanismo que se puede encontrar [en este enlace](https://nuso.org/articulo/hacia-un-futuro-transhumano/). El artículo está en español. Dejaremos aquí el abstracto de este.
# 
# > El transhumanismo es un movimiento intelectual que propone superar los límites naturales de la humanidad mediante el mejoramiento tecnológico y, eventualmente, la separación de la mente del cuerpo humano. Si bien ha sido históricamente marginal y sectario, sus planteos de medicina mejorativa, su materialismo radical, incluso sus controvertidas ideas de eugenesia, inmortalidad y singularidad adquieren creciente interés en un momento en el cual la tecnología amenaza con avanzar sobre esferas de la vida humana hasta ahora en apariencia intocables.

# En este ejercicio, extraeremos los párrafos del texto, un resumen de este, el vocabulario usado, el título, la fecha de publicación, el nombre de la revista, el nombre del autor, entre otras cosas.

# **Primero importamos las librerias necesarias**

# In[1]:


import requests
from bs4 import BeautifulSoup
import re
import nltk
import string


# Ahora, obtenemos la página como un objecto

# In[ ]:


# Obtener el código HTML del artículo
respuesta = requests.get("https://nuso.org/articulo/hacia-un-futuro-transhumano/")
pagina = BeautifulSoup(respuesta.text, 'html.parser' )


# Primeramente, vamos a obtener el nombre de la revista. el objeto `página` tiene un método que retorna el título del documento HTML

# In[ ]:


def nombre_revista(pagina):
    return pagina.title.text.split('|')[1].strip()

nombre_revista = nombre_revista(pagina)


# Podemos ver que `title` retorna, para nuestro caso específico, un array con 2 elementos: el título de la ravista y el del artícul. Ahora queremos el nombre del artículo.

# In[ ]:


def nombre_articulo(pagina):
    return pagina.title.text.split('|')[0].strip()

nombre_articulo = nombre_articulo(pagina)


# También podemos obtener el número del artículo con respecto a la revista. Esa información se encutra en la primera parte de la página web y representa cuántos artículos tenía publicado la revista, hasta el momento.

# In[ ]:


def numero_revista(pagina):
    regex = r'Nº (\d+)'
    texto_pagina = pagina.find(name='div', attrs={'class': 'section-title has-magazine'}).span.text    
    numero_revista = re.findall(regex,texto_pagina)
    return numero_revista[0]

numero_revista = numero_revista(pagina)


# Queremos obtener la fecha de la publicación del artículo

# In[ ]:


def fecha(pagina):
    texto_pagina = pagina.find(name='div', attrs={'class': 'section-title has-magazine'}).span.text
    return texto_pagina.split('/')[1].strip()

fecha = fecha(pagina)


# Queremos obtener el resumen, o el abstract, del artículo. Es la pequeña explicación sobre que trata el artículo.

# In[ ]:


def resumen_articulo(pagina):
    return pagina.find(name='div', attrs={'class':'summary'}).text.strip()

resumen_articulo = resumen_articulo(pagina)


# Finalmente, queremos el texto, o la redacción principal del artículo.

# In[ ]:


def redaccion_principal(pagina):
    parrafos = obtener_lista_de_parrafos(pagina)
    texto_articulo = ''
    for parrafo in parrafos:
        texto_articulo = texto_articulo + parrafo.text + '\n\n'
    return texto_articulo.strip()
    
redaccion_principal = redaccion_principal(pagina)


# Después de obtener todos los atributos principales del artículo, podemos hacer la "limpieza del texto". Si nosotros, por ejemplo, quisiéramos utilizar este texto para entrenar un modelo de machine learning, normalmente tenemos que eliminar los elementos del texto que no son relevantes para el modelo de machine learning. Por ejemplo, podemos eliminar las palabras de parada, las palabras como los artículos, que no brindan información importante. 
# 
# También normalmente se quiere tener el vocabulario del texto. Es decir, cuantas palabras diferentes tiene el texto.

# In[ ]:


def obtener_palabras_de_parada():
    return set( nltk.corpus.stopwords.words('spanish') + list(string.punctuation))

def obtener_vocabulario(texto_en_palabras, palabras_parada):
    return [palabra for palabra in texto_en_palabras if palabra not in palabras_parada]

palabras_parada = obtener_palabras_de_parada()
texto_en_palabras =  nltk.word_tokenize(redaccion_principal)
vocabulario = obtener_vocabulario(texto_en_palabras, palabras_parada)


# Ahora simplemente mostramos todo el código que se necesitó para realizar este ejercicio.

# In[13]:


# Funciones relativas al procesamiento del documento HTML
def nombre_revista(pagina):
    return pagina.title.text.split('|')[1].strip()

def nombre_articulo(pagina):
    return pagina.title.text.split('|')[0].strip()

def numero_revista(pagina):
    regex = r'Nº (\d+)'
    texto_pagina = pagina.find(name='div', attrs={'class': 'section-title has-magazine'}).span.text    
    numero_revista = re.findall(regex,texto_pagina)
    return numero_revista[0]

def fecha(pagina):
    texto_pagina = pagina.find(name='div', attrs={'class': 'section-title has-magazine'}).span.text
    return texto_pagina.split('/')[1].strip()

def resumen_articulo(pagina):
    return pagina.find(name='div', attrs={'class':'summary'}).text.strip()

def obtener_lista_de_parrafos(pagina):
    contenedor_texto = pagina.find(name='div', attrs={'class': 'uk-width-expand'})
    return contenedor_texto.findAll(name='p')

def redaccion_principal(pagina):
    parrafos = obtener_lista_de_parrafos(pagina)
    texto_articulo = ''
    for parrafo in parrafos:
        texto_articulo = texto_articulo + parrafo.text + '\n\n'
    return texto_articulo.strip()


# Recuperar la información en texto del artículo
nombre_revista = nombre_revista(pagina)
nombre_articulo = nombre_articulo(pagina)
numero_revista = numero_revista(pagina)
fecha = fecha(pagina)
resumen_articulo = resumen_articulo(pagina)
redaccion_principal = redaccion_principal(pagina)


# Funciones relativas a la limpieza del texto
def obtener_palabras_de_parada():
    return set( nltk.corpus.stopwords.words('spanish') + list(string.punctuation))

def obtener_vocabulario(texto_en_palabras, palabras_parada):
    return [palabra for palabra in texto_en_palabras if palabra not in palabras_parada]


# Obtener el vocabulario relevante del texto
palabras_parada = obtener_palabras_de_parada()
texto_en_palabras =  nltk.word_tokenize(redaccion_principal)
vocabulario = obtener_vocabulario(texto_en_palabras, palabras_parada)

