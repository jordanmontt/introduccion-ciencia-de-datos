# http://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk


JSON_MIME_TYPE = 'application/json'
app = Flask(__name__)


def cargar_objeto_de_archivo(ruta_archivo):
    ruta = os.path.join(ruta_archivo)
    archivo = open(ruta, "rb")
    objeto = pickle.load(open(ruta, "rb"))
    archivo.close()
    return objeto


clasificador = cargar_objeto_de_archivo('clasificador-regresion-logistica.pkl')
vocabulario_problema = cargar_objeto_de_archivo('vocabulario-problema.pkl')
palabras_de_parada = cargar_objeto_de_archivo('palabras-parada.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    solicitud_prediccion = request.get_json(force=True)
    critica = solicitud_prediccion['critica']
    # Limpiar el texto
    reg_limpiar_html = re.compile(
        '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    critica_sin_html = re.sub(reg_limpiar_html, '', critica)
    tokens = nltk.tokenize.casual.casual_tokenize(critica_sin_html, "english")
    tokens_normalizados = [token.lower() for token in tokens if token not in palabras_de_parada]
    critica_normalizada = " ".join(tokens_normalizados)
    # Obtener one hot vector
    one_hot_vector = np.zeros(len(vocabulario_problema), dtype=int)

    for token in critica_normalizada.split():
        if (token in vocabulario_problema):
            one_hot_vector[vocabulario_problema[token]] = 1

    # Predecir y retornar la respuesta
    matriz_np_critica = np.array(one_hot_vector).reshape(1,-1)
    respuesta = clasificador.predict(matriz_np_critica)
    content = json.dumps({'respuesta': respuesta[0]})
    return content, 200, {'Content-Type': JSON_MIME_TYPE}


if __name__ == '__main__':
    app.run(port=8080, debug=True)
