#!/usr/bin/env python
# coding: utf-8

# # Procesamiento de Lenguaje Natural (*NLP*)
# __[NLP](https://es.wikipedia.org/wiki/Procesamiento_de_lenguajes_naturales)__ es un campo de las ciencias de la computación, inteligencia artificial y lingüística que estudia las interacciones entre las computadoras y el lenguaje humano. Se ocupa de la formulación e investigación de mecanismos eficaces computacionalmente para la comunicación entre personas y máquinas por medio del lenguaje natural. En poca palabras, hacer que la computadora pueden entender y responder en lenguage natural.

# ### Algunas aplicaciones de *NLP*
# 
# - Extracción de palabras clave (*Keyword extraction*): identificación automática de términos importantes que mejor describan el tema de un documento.
# - Extracción de entidades (*Named-Entity Recognition*): busca localizar en el texto entidades como personas, organizaciones, lugares, expresiones de tiempo y cantidades.
# - Clasificación de texto: es asignar una categoría a un documento, esto sirve para la detección de *spam*, análisis de sentimientos, priorización de contenido, etc.
# - Resumen automático (*Text summarization*): encontrar las oraciones más informativas en un documento.
# - *Topic modeling*: es un tipo de modelo estadístico para descubrir los "topics" abstractos que ocurren en una colección de documentos. Descubre semanticas ocultas en un cuerpo de texto. Encuentra el tema en un conjunto de documentos.
# - Traducción automática (*Machine translation*): el uso de *software* para traducir texto o habla de un lenguaje natural a otro.
# 

# 1. [Vocabulario del problema](#1)
#     1. [Tokenización](#2)
#     2. [Quitar palabras de parada](#3)
#     3. [Otras tareas de limpieza que se pueden considerar](#4)
#     4. [Lematización](#5)
#     5. [Identificaciín de N-gramas](#6)
#     6. [Desambiguación lingüística (*Word Sense Disambiguation*)](#7)
#     7. [Etiquetado gramatical (*Part of Speech Tagging*)](#8)
# 2. [Representacion del texto](#9)
#     1. [One-hot encoding (tuplas de palabras)](#10)
#     2. [ TF-IDF (Term frequency, inverse document frequency)](#11)
#     3. [*Word embeddings*](#12)

# <a id="1"></a>
# # Vocabulario del problema
# En un proyecto de *NLP* se trabajan con colecciones de "documentos". Cada documento en un fragmento de texto que se debe procesar de manera individual (clasificar, representar, etc). Algunos ejemplos de documentos son: tweets, revisiones, artículos, libros, etc. El conjunto de todas la palabras que aparecen en todos los documentos constituye el **vocabulario del problema**. En muchos casos el tamaño del problema puede reducir el desempeño de la solución y es necesario reducirlo. 
# 
# Los elementos en el vocabulario se conocen como *tokens*; dependiendo del problema y las decisiones de diseño podrían no ser necesariamente una palabra. Por ejemplo, los emoticones, lemas o palabras compuestas.
# 
# En este *notebook* se muestra como realizar una de las tareas más comunes que ayudan a reducir el tamaño del vocabulario: la eliminación de palabras de parada (*stop words*).
# 

# <a id="2"></a>
# ## Tokenización
# [Tokenización](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) es el proceso de separar el texto en piezas llamadas *tokens*. Es la primera tarea/proceso en cualquier proyecto *NLP*. Es fundamental realizarla bien para no afectar la calidad de los datos de entrada en las etapas siguientes. Se recomienda usar un *tokenizer* reconocido y evitar intentar programar uno desde cero.
# 
# ![Tokenización](./img/01_tokenization.png)
# 
# A continuación, se muestra un ejemplo de tokenización sobre esta [noticia](https://www.lostiempos.com/deportes/multideportivo/20200115/olympic-albert-einstein-ucb-lpz-van-paso-firme-liga-superior). Primero debes cargar la noticia.
# 

# In[1]:


texto_noticia = """Los clubes cochabambinos de Olympic, Albert Einstein y el paceño Universidad Católica Boliviana (UCB) avanzan a paso firme y constante rumbo a la corona en la Liga Superior de voleibol, rama femenina, que se desarrolla en el coliseo Julio Borelli Vitterito de La Paz, luego de cosechar sendas victorias la noche de este martes. El ganador será representante de Bolivia en la Liga Sudamericana de Clubes 2020.

El campeón defensor del título, Olympic, superó 3-0  a su verdugo de la final de la edición 2017, el también cochabambino San Simón. La victoria para las olympiquistas fue con sets de 25-14, 25-13 y 25-18."""
texto_noticia


# ### Tokenización a nivel oraciones
# La función <font color="gray">sent_tokenize(...)</font> de <font color="gray">nltk.tokenize</font> es el tokenizador a nivel de oraciones. Si tienes problemas importando la librería [*nltk*](https://www.nltk.org/api/nltk.tokenize.html), ejecuta <font color="gray">nltk.download('punkt')</font> desde la consola de Python para descargar la librería.

# In[2]:


import nltk

oraciones_noticia = nltk.tokenize.sent_tokenize(text=texto_noticia, language='spanish')

print("Número total de oraciones de la noticia: {}".format(len(oraciones_noticia)))
print("Texto de la oracion 1: {}".format(oraciones_noticia[0]))


# ### Tokenización a nivel de palabras.
# La función <font color="gray">nltk.word_tokenize(...)</font> es el tokenizador a nivel de palabras recomendado por *NLTK*. El resultado de esta función es una lista con todas las palabras de la noticia. Internamente usa una instancia de la clase <font color="gray">TreebankWordTokenizer</font> (en la versión más reciente)
# 

# In[3]:


palabras_noticia = nltk.word_tokenize(texto_noticia)
palabras_noticia


# Debes saber que <font color="gray">word_tokenize</font>, no es la única función que permite realizar este trabajo. Puedes utilizar también <font color="gray">casual_tokenize</font>. Encuentra la diferencia =)

# In[4]:


nltk.tokenize.casual.casual_tokenize("Que buena pelicula. Gracias por invitarme :)")


# In[5]:


nltk.word_tokenize("Que buena pelicula. Gracias por invitarme :)")


# <a id="3"></a>
# ## Quitar palabras de parada
# 
# Dependiendo del lenguaje existen palabras que tienden a repetirse mucho más que otras, estas generalmente son los artículos, las preposiciones, y las conjunciones. Estas palabras suelen ser perjudiciales al momento de analizar el texto porque no aportan información relevante, es por eso que se deben quitar las palabras de parada del vocabulario del problema. Tener en cuenta que NO hay una lista universal y exhaustiva de estas palabras. Cada lenguage e incluso tipo de problema puede tener su propia lista de palabras de parada.
# 
# ![Tokenización](./img/02_remove_stopwords.png)
# 
# Puesto que estas listas pueden variar dependiendo de la librería o incluso entre versiones de la misma librería, incluir este paso puede dificultar la reproducción de los resultados en otros entornos. Veremos más adelante que hay otros mecanismos para lidiar con este tipo de palabras (*TF*, *IDF*).
# 
# Ten en cuenta tambien que quitar alguna palabra de parada como los artículos podría cambiar completamente el significado de algunas palabras compuestas.
# 
# Por ejemplo: "La Paz" nombre de un departamento de Bolivia se convierte en -> "Paz" y pierde el sentido.
# 
# Si deseas aprender más, ingresa en los siguientes enlaces:
# 
# - __[Cómo eliminar las palabras de parada usando nltk o python](https://www.pythond.com/21143/como-eliminar-las-palabras-de-parada-usando-nltk-o-python.html)__
# 
# - __[Proyecto stop-words](https://pypi.org/project/stop-words/)__
# 
# - __[Proyecto node-nltk-stopwords](https://github.com/xiamx/node-nltk-stopwords/blob/master/data/stopwords/spanish)__
# 
# - __[Stopwords @ ranks.nl](https://www.ranks.nl/stopwords)__

# Si tienes problemas importando la libreria *nltk*, ejecuta este comando <font color="gray">nltk.download('stopwords')</font>. En el ejemplo se puede ver la lista de palabras de parada que nos ofrece la libreria <font color="gray">nltk.corpus</font>

# In[6]:


import nltk

nltk.corpus.stopwords.words('spanish')


# Otra lista de palabras de parada que podrías tomar en cuenta son los signos de puntuación.

# In[7]:


import string
list(string.punctuation)


# Puedes unir ambas listas para tener todo mejor organizado

# In[8]:


palabras_de_parada = set( nltk.corpus.stopwords.words('spanish') + list(string.punctuation))
palabras_de_parada


# Finalmente solo queda eliminar las palabras de parada del texto de la noticia.

# In[9]:


palabras_noticia_sin_palabras_de_parada = [palabra for palabra in palabras_noticia if palabra not in palabras_de_parada]
palabras_noticia_sin_palabras_de_parada


# <a id="4"></a>
# ## Otras tareas de limpieza que se pueden considerar
# 
# - Unificar el case - <font color="gray">string.lower()></font>.
# - Quitar acentos (probablemente no es buena idea si el contenido está en español)
# - Procesar contracciones. I'm -> I am 
# - Quitar caracteres expeciales. Como los siguientes: #@!
# - Quitar markup. Ejemplo en contenido HTML
# - Corregir texto. Ejemplo, noooo por favoooorrr -> no por favor
# 
# Algunos enlaces de utilidad:
# - http://norvig.com/spell-correct.html
# - https://github.com/fsondej/autocorrect
# - https://github.com/MajorTal/DeepSpell

# <a id="5"></a>
# ##  Lematización
# Eliminar las variaciones de la misma palabra para tratar las variaciones como una sola entidad. (*close, closed, closing*; caminaba, caminando, caminar). Al igual que la eliminación de palabras de parada, se emplea para reducir la cantidad de elementos en el vocabulario de un problema.
# 
# Algunos enlaces que podrían servirte de ayuda:
# 
# - __[Lematización](https://es.wikipedia.org/wiki/Lematizaci%C3%B3n)__
# 
# - __[Stemmer-es, Un lematizador de español](http://stemmer-es.sourceforge.net/)__
# 
# <a id="6"></a>
# ## Identificar n-gramas
# Grupos de N palabras que están siempre juntas (Nueva York, Santa Cruz) y que deben tratarse como una sola entidad/palabra.
# 
# Algunos enlaces que podrían servirte de ayuda:
# 
# - __[Modelo bolsa de palabras](https://es.wikipedia.org/wiki/Modelo_bolsa_de_palabras)__
# 
# - __[N-grama](https://es.wikipedia.org/wiki/N-grama)__
# 
# <a id="7"></a>
# ## Desambiguación lingüística (*Word Sense Disambiguation*)
# Asignar significado en base al contexto. ¿Con qué sentido se usa una palabra? En el caso de la __[polisemia](https://es.wikipedia.org/wiki/Polisemia)__, ¿cuál de los significados es el más apropiado dado el contexto?
# 
# Ejemplos:
# 
# - Placeres de la carne.
# - La carne está sabrosa.
# 
# 
# - Puso dos velas a San Pedro.
# - Los egipcios fueron los primeros constructores de barcos de vela de los que se tiene noticia.
# 
# El desarrollo de algoritmos para reproducir esta capacidad humana (desambiguar el significado) a menudo puede ser una  __[tarea muy difícil](https://es.wikipedia.org/wiki/Problema_no_resuelto)__. En la frase "La carne está sabrosa" hay también cierto contenido implícito: se asume que estamos hablando de carne cocida.
# 
# Algunos enlaces que podrían servirte de ayuda:
# 
# - __[Desambiguación lingüística](https://es.wikipedia.org/wiki/Desambiguaci%C3%B3n_ling%C3%BC%C3%ADstica)__
# 
# 
# Recursos lingüísticos
# 
# - __[WordNet](https://es.wikipedia.org/wiki/WordNet)__
# 
# - __[Spanish WordNet 3.0](http://timm.ujaen.es/recursos/spanish-wordnet-3-0/)__
# 
# 
# Estrategias 
# 
# - __[Algoritmo Lesk](https://en.wikipedia.org/wiki/Lesk_algorithm)__
# 
# - __[Desambiguación del Sentido de las Palabras](http://dpinto.cs.buap.mx/pln/Autumn2010/wsd.pdf)__
# 
# - __[Estudio sobre métodos tipo Lesk usados para la desambiguación de sentidos de palabras](https://pdfs.semanticscholar.org/cd5f/5dd14c126325a81280407ddc2616f3704fca.pdf)__
# 
# - __[Supervised Word Sense Disambiguation: Facing Current Challenges](http://www.sepln.org/sites/default/files/monografia/archivos/2018-10/monografiaDavid.pdf)__
# 
# <a id="8"></a>
# ##  Etiquetado gramatical (*Part of Speech Tagging*)
# Como parte de la desambiguación, se suele realizar la tarea de asignar una categoría a cada palabra: sujeto, nombre, verbo, adjetivo. 
# 
# Algunos enlaces de ayuda:
# 
# - __[Etiquetado gramatical](https://es.wikipedia.org/wiki/Etiquetado_gramatical)__
# 
# - __[Using Wikicorpus & NLTK to build a Spanish part-of-speech tagger](https://www.cnts.ua.ac.be/pages/using-wikicorpus-nltk-to-build-a-spanish-part-of-speech-tagger)__
# 
# - __[Choosing a Spanish Part-of-Speech tagger for a lexically sensitive task](https://www.researchgate.net/publication/282828110_Choosing_a_Spanish_Part-of-Speech_tagger_for_a_lexically_sensitive_task)__

# <a id="9"></a>
# 
# # Representación del texto
# 
# Dependiendo del tipo de problema *NLP*, el texto deberá transformarse en una representación adecuada para las herramientas y algoritmos empleandos para abordar el problema. Una de las más usadas es crear representaciones numéricas. Esencialmente, se trata de convertir texto en un vector/arreglo de números.
# 
# Los valores en el arreglo pueden ser por ejemplo: la frequencia del *tóken* en el documento, el código en un *word embedding*, o la métrica *TF-IDF*. 
# 
# Hay escenarios en los cuales estos vectores pueden tener un gran tamaño, en estos casos se recomienda emplear alguna estrategia para reducir la dimensionalidad de los vectores (Ej. *feature hashing*, *locality sensitive hashing*).
# 
# Para su mejor comprensión, realizaremos la representación sobre la base de esta [noticia](https://www.lostiempos.com/deportes/multideportivo/20200115/olympic-albert-einstein-ucb-lpz-van-paso-firme-liga-superior)

# Primero debemos obtener el texto de la noticia.

# In[10]:


texto_noticia = """Los clubes cochabambinos de Olympic, Albert Einstein y el paceño Universidad Católica Boliviana (UCB) avanzan a paso firme y constante rumbo a la corona en la Liga Superior de voleibol, rama femenina, que se desarrolla en el coliseo Julio Borelli Vitterito de La Paz, luego de cosechar sendas victorias la noche de este martes. El ganador será representante de Bolivia en la Liga Sudamericana de Clubes 2020.

El campeón defensor del título, Olympic, superó 3-0  a su verdugo de la final de la edición 2017, el también cochabambino San Simón. La victoria para las olympiquistas fue con sets de 25-14, 25-13 y 25-14."""
texto_noticia


# Ahora debemos importar las librerias correspondientes y generar la lista de *stop words* en español.

# In[11]:


import nltk
import string
import numpy as np
import pandas as pd
import collections
import math

# nltk.download('stopwords') # Si no está ya descargado

palabras_de_parada_espanol = set( nltk.corpus.stopwords.words('spanish') + list(string.punctuation))


# Luego proceguimos a tokenizar el texto a nivel oración.

# In[12]:


# nltk.download('punkt') # Si no está ya presente

oraciones_noticia = nltk.tokenize.sent_tokenize(text=texto_noticia, language='spanish')
oraciones_noticia[0]


# Creamos una funcion para eliminar las palabras de parada de una oración.

# In[13]:


def normalizar_oraciones(oraciones, language='spanish'):
    tokens = nltk.word_tokenize(oraciones,language)
    tokens_filtrados = [token.lower() for token in tokens if token not in palabras_de_parada_espanol]
    return " ".join(tokens_filtrados)


# Aplicamos la función a todo el cuerpo del texto.

# In[14]:


def normalizar_cuerpo_texto(oraciones, language='spanish'):
    return np.array([normalizar_oraciones(oracion, language) for oracion in oraciones])

cuerpo_texto_normalizado = normalizar_cuerpo_texto(oraciones_noticia)
cuerpo_texto_normalizado


# Muy bien, ahora obtendremos el vocabulario del problema con el siguiente código.

# In[15]:


def obtener_vocabulario_problema(cuerpo_texto_normalizado):
    vocabulario_problema = [] 
    for oracion in cuerpo_texto_normalizado:
        vocabulario_problema.extend(oracion.split())  
    return vocabulario_problema


# Con la siguiente función, tendremos cada palabra del vocabulario con su posición.

# In[16]:


def obtener_vocabulario_problema_y_posicion(vocabulario_del_problema_ordenado):
    token_y_su_posicion = {}
    for i, token in enumerate(vocabulario_del_problema_ordenado):
        token_y_su_posicion[token] = i
    return token_y_su_posicion

vocabulario_problema = obtener_vocabulario_problema(cuerpo_texto_normalizado)
vocabulario_problema_ordenado = sorted(set(vocabulario_problema))
vocabulario_problema_y_su_posicion = obtener_vocabulario_problema_y_posicion(vocabulario_problema_ordenado)
print("Cantidad de palabras:", len(vocabulario_problema_y_su_posicion))
vocabulario_problema_y_su_posicion


# <a id="10"></a>
# 
# ### One-hot encoding (tuplas de palabras)
# 
# Los *feature vectors* se utilizan para representar características simbólicas o númericas; llamados *features*, de un objeto de una manera matemática y fácilmente analizable.
# 
# Se crea un vocabulario con todas la palabras de todos los documentos. Y cada documento se representa como una arreglo donde se indica la presencia o ausencia de una palabra en el documento.
# 
# ![one-hot-encoding](./img/one-hot-encoding.png)
# 
# - Cada documento se representa con uno arreglo tan grande como todo el vocabulario! :(
# - Se pierde el órden y la frequencia de las palabras.
# - No captura relaciones entre palabras.
# - Su mayor ventaja es la simplicidad.

# In[17]:


cuerpo_texto_normalizado[2]


# In[18]:


def vector_one_hot(oracion, vocabulario_problema_y_su_posicion):
    vector = np.zeros(len(vocabulario_problema_y_su_posicion),dtype=int)
    for token in oracion.split():
        vector[vocabulario_problema_y_su_posicion[token]] = 1
    return vector

vector_one_hot(cuerpo_texto_normalizado[2], vocabulario_problema_y_su_posicion)


# <a id="11"></a>
# 
# ### TF-IDF (Term frequency, inverse document frequency)
# 
# **Conteos**
# 
# El vector de cada documento contiene la cantidad de veces que aparece una palabra (no solo se marca su presencia/ausencia)
# 
# **TF-IDF (Term frequency, inverse document frequency)**
# 
# *TF*: La frecuencia de una palabra en un documento.
# 
# *IDF*: Mientras en más documentos aparece menos significativa es la palabra en los documentos en las aparece.
# 
# Captura la frequencia de las palabras en cada documento y en el contenido formado por el conjunto de los documentos.
# 
# El valor de *TF-IDF* para cada palabra aumenta por su frecuencia en un documento pero disminuye si al mismo tiempo aparece 
# todo el conjunto de documentos (es un término común).
# 
# La idea es capturar la importancia de las palabras en los documentos. No captura las relaciones entre las palabras.
# 

# #### TF  (Term Frequency)

# In[19]:


cuerpo_texto_normalizado


# In[20]:


vocabulario_problema_y_su_posicion


# Contamos la frecuencia de cada *token*.

# In[21]:


tokens_y_su_frecuencia = collections.Counter(cuerpo_texto_normalizado[3].split())
tokens_y_su_frecuencia


# Puedes contar también de esta forma:

# In[22]:


frecuencia = collections.Counter(['a','b','c','a'])
frecuencia


# Con el siguiente código puedes obtener todos los *tokens*

# In[23]:


from heapq import nlargest

nlargest(10, frecuencia, key=frecuencia.get)


# La siguiente función anota la frecuencia de cada palabra en el vector *one hot*

# In[24]:


def obtener_vector_frecuencia_termino(oracion, vocabulario_problema_y_su_posicion):
    vector = np.zeros(len(vocabulario_problema_y_su_posicion),dtype=int)
    tokens_y_su_frecuencia = collections.Counter(oracion.split());
    for token, frecuencia in tokens_y_su_frecuencia.items():
        vector[vocabulario_problema_y_su_posicion[token]] = frecuencia
    return vector

obtener_vector_frecuencia_termino(cuerpo_texto_normalizado[3], vocabulario_problema_y_su_posicion)


# ***Evite escribir código, cuando se puede hacer con librerias***
# 
# Un enlace de ayuda:
# __[Scikit-Learn Design Principles](https://towardsdatascience.com/scikit-learn-design-principles-d1371958059b)__

# In[25]:


simple_cuerpo_texto_normalizado = np.array(['p1 co','p2 co co'])
simple_cuerpo_texto_normalizado


# La funcion <font color="gray">CountVectorizer()</font> implementa <font color="gray">transform</font>

# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_matris = cv.fit_transform(simple_cuerpo_texto_normalizado)
cv_matris


# Algunos ejemplos más:

# In[27]:


print("Matriz disperza:")
print(cv_matris)
print("Vecto np regular:")
print(cv_matris.toarray())
print("Vocabulario:")
print(cv.vocabulary_)
print("Variables:")
print(cv.get_feature_names())
print("one-hot encoding:")
print(np.sign(cv_matris.toarray()))


# Puedes también utilizar los *dataframes*

# In[28]:


pd.DataFrame(cv_matris.toarray(), columns=cv.get_feature_names())


# Ahora unos ejemplos para bi-gramas.

# In[29]:


simple_cuerpo_texto_normalizado


# In[30]:


bv = CountVectorizer(ngram_range=(1,2))
cv_matris_con_bigramas = bv.fit_transform(simple_cuerpo_texto_normalizado)
print("Matris dispersa:")
print(cv_matris_con_bigramas)
print("Vector np regular:")
print(cv_matris_con_bigramas.toarray())
print("Variables:")
print(bv.get_feature_names())


# Puedes también utilizar *dataframes*

# In[31]:


pd.DataFrame(cv_matris_con_bigramas.toarray(), columns=bv.get_feature_names())


# #### IDF  (Inverse Document Frequency)

# idf(t) = log(cantidad-documentos/cantidad-documentos-con-el-termino)
# 
# Min = 1; A valores más alejados de 1, mayor IDF
# 
# Interpretación de resultados

# In[32]:


simple_cuerpo_texto_normalizado = np.array(['p1 co','p2 co co'])
simple_cuerpo_texto_normalizado


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfv = TfidfVectorizer(norm='l2')
tfIdfv.fit(simple_cuerpo_texto_normalizado)


# Fijate lo que pasa con el valor idf para las palabras que aparecen en muchos documentos. Debido a cómo sckit-learn hace los cálculos, no hay IDF = 0, el valor mínimo es 1

# In[34]:


dict(zip(tfIdfv.get_feature_names(),tfIdfv.idf_ ))


# In[35]:


simple_cuerpo_texto_normalizado = np.array(['p1 co co','p2 co'])
tfIdfv.fit(simple_cuerpo_texto_normalizado)


# Este *score*/indicador aumenta con la frequencia de la palabra en el documento, pero disminuye cuando la palabra se hace muy común (baja su relevancia)

# In[36]:


tt_matris = tfIdfv.transform(simple_cuerpo_texto_normalizado)
print("Vector np regular:")
print(tt_matris.toarray())
print("Variables:")
print(tfIdfv.get_feature_names())


# <a id="12"></a>
# 
# ### Word embeddings
# 
# Asociación de palabras con códigos numéricos que capturan su similitud semánticas. Por ejemplo, Londres tendrá un valor numérico cercano a París porque ambas palabras tiene significado parecido (ambas son ciudades importantes de europa); de la misma manera, la distancia entre las palabras "varón" y "mujer" sería similar a la distancia que existe entre "rey" y "reina".
# 
# Son generadas a partir de grandes cuerpos de texto por algoritmos basados en redes neuronales, reducción de la dimensionalidad de matrices de co-ocurrencia y modelos probabilísticos.
# 
# Algunos enlaces de ayuda:
# - https://en.wikipedia.org/wiki/Word2vec
# 
# - https://github.com/aitoralmeida/spanish_word2vec
