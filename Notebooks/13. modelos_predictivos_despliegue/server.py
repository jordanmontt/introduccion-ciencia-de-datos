#http://flask.palletsprojects.com/en/1.1.x/
#http://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def echo():
    data = request.get_json(force=True)
    print(data)
    print(type(data))
    return jsonify(data)

#Python asigna el valor __main__ a la variable __name__ cuando se ejecuta en modo standalone
if __name__ == '__main__':
    app.run(port=8080, debug=True)