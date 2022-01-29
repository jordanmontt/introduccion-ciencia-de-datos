# Introducción a la Ciencia de Datos
Este es un breve curso introductorio en Python a la Ciencia de Datos.

Sitio web con el contenido: [jordanmontt.github.io/introduccion-ciencia-de-datos](https://jordanmontt.github.io/introduccion-ciencia-de-datos/intro.html)

Para agregar cambios, se debe: 

1. Instalar los siguientes paquetes:

```
pip install -U jupyter-book
pip install ghp-import
```

2. Compilar de nuevo el código ejecutando el siguiente comando en el root del repositorio:

```
jupyter-book build Web\ Application/
```

3. Deployar los cambios ejecutando el siguiente comando:

```
ghp-import -n -p -f Web\ Application/_build/html
```
