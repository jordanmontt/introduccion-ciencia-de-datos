#!/usr/bin/env python
# coding: utf-8

# # Web Scraping
# 
# __[Web Scraping](https://www.imperva.com/learn/application-security/web-scraping-attack/)__ es el proceso de utilizar bots para extraer contenido y datos de un sitio web. En este proceso, se extrae el código HTML y los datos almacenados en una base de datos. Web Scraping permite replicar el contenido completo de un sitio web en cualquier lugar. Muchos negocios digitales que dependen de la recolección de datos, utilizan este proceso.
# 
# A continuación, se mencionan algunos de los casos aplicables de esta técnica:
# - Clasificación de contenido, a tráves de la extracción de datos de multiples sitios web.
# - Comparación de precios y descripciones de productos sobre la base de sitios web que ofrecen productos.
# - Análisis de sentimiento a tráves de la extracción de datos de foros y redes sociales.
# 
# Web Scraping es también utilizado para propósitos ilegales como ser: la subvaloración de precios y el robo de contenido protegido por derechos de autor. Esto podría provocar pérdidas financieras, especialmente si es una empresa que depende fuertemente de modelos de precios competitivos u ofertas en la distribución de contenido. Normalmente, las páginas tiene un archivo robots.txt en el cual especifican si es permitido o no extraer datos de ellas.
# 
# __Nota__: evita realizar este proceso si previamente no cuentas con autorización del autor.
# 
# Si quieres saber más, te dejamos abajo unos enlaces que podrían interesarte.
# 
# 
# - __[Estándar de exclusión de robots](https://es.wikipedia.org/wiki/Est%C3%A1ndar_de_exclusi%C3%B3n_de_robots)__
# 
# - __[5 Things You Need to Know Before Scraping Data From Facebook](https://www.octoparse.com/blog/5-things-you-need-to-know-before-scraping-data-from-facebook)__

# 1. [Extraer el código HTML](#1)
# 2. [Realizar búsquedas](#2)
# 3. [Experimentar estrategias para extraer datos](#3)
# 4. [Crear funciones y procesar los elementos](#4)
# 5. [Almacenar los datos en archivos](#5)

# <a id="1"></a>
# ## Extraer el código HTML
# 
# Para explicar el proceso de Web Scraping, analizaremos la página web [Auto MPG web page](http://localhost:8000/auto_mpg.html). Esa página web es una página que fue creada para este tutorial. Recolecta los datos open source del dataset [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg). Puedes crear un pequeño servidor que retorne página [Auto MPG web page](http://localhost:8000/auto_mpg.html). Puedes cambiar el puerto si es diferente al 8000.
# 
# Las columnas del dataset son las siguientes: nombre, cilindros, peso, año, territorio, aceleración, millas por galón, caballos de potencia y desplazamiento.
# 
# Ejecuta la siguientes líneas de código:

# In[1]:


import requests

respuesta = requests.get("http://localhost:8000/auto_mpg.html")


# La biblioteca BeautifulSoup de bs4 nos permitirá análizar documentos HTML. 
# 
# Ejecuta las siguientes líneas de código para utilizar esta librería:

# In[ ]:


from bs4 import BeautifulSoup
pagina = BeautifulSoup(respuesta.text, "html.parser" )


# Vizualiza si extrajiste bien el código HTML

# In[ ]:


print(pagina.prettify())


# Puedes acceder directamente al body del html utilizando pagina.body

# In[ ]:


print(pagina.body)


# <a id="2"></a>
# ## Realizar búsquedas
# Puedes utilizar la función .find y .find_all para encontrar los elementos que deseas

# In[ ]:


print(pagina.body.find(name="div", attrs={"id":"car-1"}))
print(len(pagina.body.find_all(name="div", attrs={"class":"car_block"})))
print(pagina.body.find_all(name="div", attrs={"class":"car_block"})[0])


# Puedes acceder a los atributos de cada elemento como en el siguiente ejemplo.

# In[ ]:


car_divs = pagina.body.find_all(name="div", attrs={"class":"car_block"})
div = car_divs[0]
div['class']


# <a id="3"></a>
# ## Experimentar estrategias para extraer datos
# Si deseas extraer únicamente el contenido sin elementos HTML, realiza lo siguiente.

# In[ ]:


print(div.text)


# Puedes crear un generador (similar a un iterable) de la siguiente forma.

# In[ ]:


print(div.stripped_strings)
print(list(div.stripped_strings))


# Puedes utilizar funciones poprias del lenguaje para obtener un mejor resultado de los datos que deseas obtener.

# In[ ]:


div.find("span", attrs={"class":"mpg"}).text
div.find("span", attrs={"class":"mpg"}).text.split(" ")[0]


# Incluso puedes utilizar expresiones regulares si así lo deseas.

# In[ ]:


import re
re.findall(r'.* (\d+.\d+) cubic inches', div.text)[0]


# <a id="4"></a>
# ## Crear funciones y procesar los elementos
# Finalmente, te recomendamos crear funciones para cada item y procesar los elementos que contienen los datos.
# 
# A continuación se muestra un ejemplo:

# In[ ]:


import csv
import re
import requests
from bs4 import BeautifulSoup

def extraer_desplazamiento(div_automovil_text):
    texto_de_desplazamiento = re.findall(r'.* (\d+.\d+) cubic inches', div_automovil_text)[0]
    return float(texto_de_desplazamiento)

def extraer_caballos_potencia(div_automovil):
    texto_caballo_potencia = div_automovil.find('span', class_='horsepower').text
    try:
        texto_caballo_potencia = float(texto_caballo_potencia)
    except ValueError:
        texto_caballo_potencia = "NULL"
    return texto_caballo_potencia

def extraer_mpg(div_automovil):
    texto_de_mpg = div_automovil.find("span", attrs={"class":"mpg"}).text
    try:
        mpg = float(texto_de_mpg.split(' ')[0])
    except ValueError:
        mpg = "NULL"
    return mpg

def extraer_aceleracion(div_automovil):
    return float(div_automovil.find('span', class_='acceleration').text)

def extraer_territorio_y_anio(div_automovil):
    texto_de_from = div_automovil.find("span", attrs={"class":"from"}).text
    anio, territorio = texto_de_from.strip('()').split(',')
    anio = int(anio.strip())
    territorio = territorio.strip()
    return territorio, anio

def extraer_peso(div_automovil):
    texto_de_peso = div_automovil.find('span', class_='weight').text
    return int(texto_de_peso.replace(',', ''))

def extraer_cilindros(div_automovil):
    return int(div_automovil.find('span', class_='cylinders').text)

def extraer_nombre(div_automovil):
    return div_automovil.find('span', class_='car_name').text

def extraer_datos(div_automovil):
    fila = {}
    fila["nombre"] = extraer_nombre(div_automovil)
    fila["cilindros"] = extraer_cilindros(div_automovil)
    fila["peso"] = extraer_peso(div_automovil)
    fila["territorio"], fila["anio"] = extraer_territorio_y_anio(div_automovil)
    fila["aceleracion"] = extraer_aceleracion(div_automovil)
    fila["mpg"] = extraer_mpg(div_automovil)
    fila["caballos_potencia"] = extraer_caballos_potencia(div_automovil)
    fila["desplazamiento"] = extraer_desplazamiento(div_automovil.text)
    return fila

def extraer_datos_automoviles(pagina):
    divs_automoviles = pagina.body.find_all(name="div", attrs={"class":"car_block"})
    return list(map(extraer_datos, divs_automoviles))

        
respuesta = requests.get("http://localhost:8000/auto_mpg.html")
pagina = BeautifulSoup(respuesta.text, "html.parser" )
datos_automoviles = extraer_datos_automoviles(pagina)

print(f"Se ha extraido {len(datos_automoviles)} filas")
print(datos_automoviles[1])
print(datos_automoviles[-1])


# <a id="5"></a>
# ## Almacenar los datos en archivos
# Puedes almacenar los datos extraídos en un archivo de tu preferencia

# In[ ]:


def almacenar_datos_en_un_archivo_csv(datos_automoviles):
    with open("datos_automoviles1.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=datos_automoviles[0].keys())
        writer.writeheader()
        writer.writerows(datos_automoviles)
almacenar_datos_en_un_archivo_csv(datos_automoviles)

