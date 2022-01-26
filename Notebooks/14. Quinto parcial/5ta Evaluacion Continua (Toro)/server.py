import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.externals import joblib


ruta_archivo_clasificador = os.path.join( "clasificador","Classifier.pkl")
model_columns = pickle.load(open(ruta_archivo_clasificador, "rb"))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def helo():
    return 'Bienvenido al clasificador'

@app.route('/predict', methods=['POST'])
def predict():
    #all kinds of error checking should go here
    data = request.get_json(force=True)
    #convert our json to a numpy array
    predict_request = [data['Pregnancies'],data['Glucose'],data['BloodPressure'],data['SkinThickness'],data['Insulin'],data['BMI'],data['DiabetesPedigreeFunction'],data['Age']] 
    predict_request = np.array(list(predict_request)).reshape(1,-1)
    print('====================')
    print(type(predict))
    #np array goes into random forest, prediction comes out
    y_hat = model_columns.predict(predict_request)
    print(y_hat)
    #return our prediction
    output = {'Diabetes?': int(y_hat[0])}
    print(output)
    return output
    
if __name__ == '__main__':
    clf = joblib.load('model.pkl')
    app.run(port=8000, debug=True)