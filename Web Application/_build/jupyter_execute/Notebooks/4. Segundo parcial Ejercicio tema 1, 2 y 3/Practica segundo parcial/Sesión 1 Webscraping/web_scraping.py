#!/usr/bin/env python
# coding: utf-8

# Web Scraping: extraer datos del contenido HTML de un sitio web
# 
# Realizar ese proceso contenido no autorizado por el autor es ilegal!!

# <a href="https://es.wikipedia.org/wiki/Web_scraping" target="_blank">Web scraping</a></br>
# <a href="https://es.wikipedia.org/wiki/Est%C3%A1ndar_de_exclusi%C3%B3n_de_robots" target="_blank">Estándar de exclusión de robots</a></br>
# <a href="https://www.imperva.com/learn/application-security/web-scraping-attack/" target="_blank">What is web scraping</a></br>
# <a href="https://www.octoparse.com/blog/5-things-you-need-to-know-before-scraping-data-from-facebook" target="_blank">5 Things You Need to Know Before Scraping Data From Facebook</a></br>
# <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" target="_blank">Beautiful Soup Documentation</a></br>

# In[1]:


#https://archive.ics.uci.edu/ml/datasets/auto+mpg
#Auto MPG Data Set
#Inciar el servidor web localmente: 
#python -m http.server
import requests
respuesta = requests.get("http://localhost:8000/auto_mpg.html")


# In[3]:


#Importar la clase BeautifulSoup del m
from bs4 import BeautifulSoup
pagina = BeautifulSoup(respuesta.text, "html.parser" )
#pagina?
#print(pagina.prettify())
#Demo. Chrome Developer Tools: Explorar la estructura de la página para identificar los elementos que contienen los datos


# In[10]:


#print(pagina.body)
#print(pagina.body.find(name="div", attrs={"id":"car-1"}())
#print(len(pagina.body.find_all(name="div", attrs={"class":"car_block"})))
#print(pagina.body.find_all(name="div", attrs={"class":"car_block"})[0])
car_divs = pagina.body.find_all(name="div", attrs={"class":"car_block"})
div = car_divs[0]
#print(div.prettify())
#print(div)
#print(div.text) #Sin elementos HTML  :)
#print(div.stripped_strings) #generador, similar a un iterable
#print(list(div.stripped_strings))


# In[69]:


#div.find_all("span")
div.find("span", attrs={"class":"mpg"}).text
div.find("span", attrs={"class":"mpg"}).text.split(" ")[0]


# In[11]:


print(div.prettify())
import re
re.findall(r'.* (\d+.\d+) cubic inches', div.text)[0]
#Continuar experimentando estrategias para extraer datos
#Finalmente, crear funciones para cada item de datos y procesar los elementos que contienen loa datos


# In[4]:


#Proceso extraccion
import csv
import re
import requests
from bs4 import BeautifulSoup

def extraer_territorio_anio(div_automovil):
    str_from = div_automovil.find("span", attrs={"class":"from"}).text
    anio, territorio = str_from.strip('()').split(',')
    anio = int(anio.strip())
    territorio = territorio.strip()
    return territorio, anio

def extraer_mpg(div_automovil):
    mpg_str = div_automovil.find("span", attrs={"class":"mpg"}).text
    try:
        mpg = float(mpg_str.split(' ')[0])
    except ValueError:
        mpg = "NULL"
    return mpg

def extraer_caballos_potencia(div_automovil):
    caballos_potencia_str = div_automovil.find('span', class_='horsepower').text
    try:
        caballos_potencia = float(caballos_potencia_str)
    except ValueError:
        caballos_potencia = "NULL"
    return caballos_potencia

def extraer_desplazamiento(div_automovil_text):
    str_desplazamiento = re.findall(r'.* (\d+.\d+) cubic inches', div_automovil_text)[0]
    desplazamiento = float(str_desplazamiento)
    return desplazamiento  

def extraer_datos(div_automovil):
    nombre = div_automovil.find('span', class_='car_name').text
    str_cilindros = div_automovil.find('span', class_='cylinders').text
    cilindros = int(str_cilindros)
    str_peso = div_automovil.find('span', class_='weight').text
    peso = int(str_peso.replace(',', ''))
    territorio, anio = extraer_territorio_anio(div_automovil)
    aceleracion = float(div_automovil.find('span', class_='acceleration').text)
    mpg = extraer_mpg(div_automovil)
    caballos_potencia = extraer_caballos_potencia(div_automovil)
    desplazamiento = extraer_desplazamiento(div_automovil.text)
    fila = dict(nombre=nombre,
               cilindros=cilindros,
               peso=peso,
               anio=anio,
               territorio=territorio,
               aceleracion=aceleracion,
               mpg=mpg,
               caballos_potencia=caballos_potencia,
               desplazamiento=desplazamiento)
    return fila

def extraer_datos_automoviles(pagina):
    """Extract information from repeated divisions"""
    divs_automoviles = pagina.body.find_all(name="div", attrs={"class":"car_block"})
    filas = []
    for div_automovil in divs_automoviles:
        fila = extraer_datos(div_automovil)
        filas.append(fila)
    print(f"Se extraido {len(filas)} filas")
    print(filas[0])
    print(filas[-1])

    with open("datos_automoviles.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=fila.keys())
        writer.writeheader()
        writer.writerows(filas)
        
respuesta = requests.get("http://localhost:8000/auto_mpg.html")
pagina = BeautifulSoup(respuesta.text, "html.parser" )
extraer_datos_automoviles(pagina)

