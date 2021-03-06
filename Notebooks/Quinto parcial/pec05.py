# http://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

classifier_filepath = os.path.join("classifier.pkl")
classifier_file = open(classifier_filepath, "rb")
classifier = pickle.load(open(classifier_filepath, "rb"))
classifier_file.close()

# Desactiva el API /predict del clasificador.
# retorna {"message": "/predict disabled"}, 200 OK
@app.route('/disable', methods=['GET'])
def disable():
    global ACTIVATED
    ACTIVATED = False
    return {'message': '/predict disabled'}, 200

# Activa el API /predict del clasificador.
# retorna {"message": "/predict enabled"}, 200 OK
@app.route('/enable', methods=['GET'])
def enable():
    global ACTIVATED
    ACTIVATED = True
    return {'message': '/predict enabled'}, 200

# Entrena el modelo con los nuevos hyper-parámetros y retorna la nueva exactitud. Por ejemplo, {"accuracy": 0.81}, 200 OK
# Se pueden enviar los siguiente hyper-parámetros: { "n_estimators": 10, "criterion": "gini", "max_depth": 7 }
# "criterion" puede ser "gini" o "entropy", "n_estimators" y "max_depth" son un número entero positivo
# Unicamente "max_depth" es opcional en cuyo caso se deberá emplear None. Si los otros hyper-parámetros no están presentes se retorna:
# {"message": "missing hyper-parameter"}, 404 BAD REQUEST
# Finalmente, sólo se puede ejecutar este endpoint después de ejecutar GET /disable. En otro caso retorna {"message": "can not reset an enabled classifier"}, 400 BAD REQUEST
@app.route('/reset', methods=['POST'])
def reset():
    if ACTIVATED:
        return {"message": "can not reset an enabled classifier"}, 400
    json_request = request.get_json(force=True)
    if 'criterion' not in json_request or 'n_estimators' not in json_request:
        return {"message": "missing hyper-parameter"}, 400

    classifier.n_estimators = json_request.get('n_estimators')
    classifier.criterion = json_request.get('criterion')
    classifier.max_depth = json_request.get('max_depth')
 
    df = pd.read_csv(os.path.join("diabetes.csv"))
    feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age',
                    'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
    X = df[feature_cols]
    Y = df["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1)

    classifier.fit(X_train, Y_train)
    return {'accuracy': accuracy_score(Y_test, classifier.predict(X_test))}, 200

# Recibe una lista de observaciones y retorna la clasificación para cada una de ellas.
# Los valores en cada observación se corresponden con la siguientes variables:
#['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
# Por ejemplo: para estas observaciones:
# [
#	[7,135,26.0,51,136,74,0.647],
#	[9,175,34.2,36,112,82,0.260]
# ]
@app.route('/predict', methods=['POST'])
def predict():
    if not ACTIVATED:
        return {"message": "classifier is not enabled"}, 400
    predict_request = request.get_json(force=True)
    predict_response = classifier.predict(predict_request)
    return {'cases': predict_request,
            'diabetes': predict_response.tolist()}


if __name__ == '__main__':
    app.run(port=8080, debug=True)
