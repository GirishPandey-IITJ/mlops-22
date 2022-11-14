from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
path = "dt_max_depth=20.joblib"
alg = load(path)
@app.route("/predict" , methods = ['POST'])

def predict():
    image = request.json['image']
    print("loading complete")
    prd = alg.predict([image])
    return {"y_predicted" : int(prd[0])}