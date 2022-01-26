#http://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

app = Flask(__name__)

classifier_filepath = os.path.join("classifier.pkl")
classifier_file = open(classifier_filepath, "rb")
classifier = pickle.load(open(classifier_filepath, "rb"))
classifier_file.close()
cache = {}
cache['enabled'] = True

#1
#Desactiva el API /predict del clasificador.
#retorna {"message": "/predict disabled"}, 200 OK
@app.route('/disable', methods=['GET'])
def disable():
    cache['enabled']=False
    print( cache['enabled'])
    return {"message":"/predict disabled"},200

#2
#Activa el API /predict del clasificador.
#retorna {"message": "/predict enabled"}, 200 OK
@app.route('/enable', methods=['GET'])
def enable():
    enabled = True
    print(enabled)
    return {"message":"/predict enabled"},200

#3
#Entrena el modelo con los nuevos hyper-parámetros y retorna la nueva exactitud. Por ejemplo, {"accuracy": 0.81}, 200 OK
#Se pueden enviar los siguiente hyper-parámetros: { "n_estimators": 10, "criterion": "gini", "max_depth": 7 }
#"criterion" puede ser "gini" o "entropy", "n_estimators" y "max_depth" son un número entero positivo
#Unicamente "max_depth" es opcional en cuyo caso se deberá emplear None. Si los otros hyper-parámetros no están presentes se deberá retornar
#{"message": "missing hyper-parameter"}, 404 BAD REQUEST
#Finalmente, sólo se puede ejecutar este endpoint después de ejecutar GET /disable. En otro caso retorna {"message": "can not reset an enabled classifier"}, 400 BAD REQUEST 
@app.route('/reset', methods=['POST'])
def reset():
    if  cache['enabled'] == True:
        response = {"message": "can not reset an enabled classifier"}, 400
    else:
        data = request.get_json(force=True)
        if "criterion" not in data or "n_estimators" not in data:
            response = {"message": "missing hyper-parameter"}, 404
        else:
            df = pd.read_csv(os.path.join("diabetes.csv"))
            feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
            X = df[feature_cols] # Features
            y = df["Outcome"] # Target variable
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            classifier.n_estimators = data["n_estimators"]
            classifier.criterion = data["criterion"]
            classifier.max_depth = data["max_depth"]
            classifier.fit(X_train,y_train)
            response = {"accuracy":accuracy_score(y_test, classifier.predict(X_test))}, 200
    return response


#3
#Recibe una lista de observaciones y retorna la clasificación para cada una de ellas.
#Los valores en cada observación se corresponden con la siguientes variables: 
#['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
#Por ejemplo: para estas observaciones: 
#[
#	[7,135,26.0,51,136,74,0.647],
#	[9,175,34.2,36,112,82,0.260]
#]
#La respuesta puede ser:
#{
#    "cases": [
#        [7,135,26.0,51,136,74,0.647],
#        [9,175,34.2,36,112,82,0.260]
#   ],
#    "diabetes": [0,0]
#}
#Finalmente, sólo se puede ejecutar este endpoint después de ejecutar este API esta activado (enabled). En otro caso retorna {"message": "classifier is not enabled"}, 400 BAD REQUEST 
@app.route('/predict', methods=['POST'])
def predict():
    if  cache['enabled'] == False:
        response =  {"message": "classifier is not enabled"}, 400
    else:
        data = request.get_json(force=True)
        output = []
        for case in data['cases']:
            predict_request = [case['Pregnancies'],case['Glucose'],case['BloodPressure'],case['Insulin'],case['BMI'],case['DiabetesPedigreeFunction'],case['Age']] 
            predict_request = np.array(list(predict_request)).reshape(1,-1)
            prediction = classifier.predict(predict_request)
            output.append({'Diabetes': int(prediction[0])})
        response = jsonify(output),200
    return response
    

if __name__ == '__main__':
    app.run(port=8080, debug=True)