import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)

app = application

# Import Models
ridge_model = pickle.load(open('Model\\Ridge.pkl','rb'))
Scalar = pickle.load(open('Model\\Scalar.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData', methods=["GET","POST"])
def predict_data():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_scaled_data = Scalar.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        result = ridge_model.predict(new_scaled_data)
        
        return render_template('home.html', result=round(result[0],2))
        
    else:
        return render_template('home.html')
    

if(__name__=='__main__'):
    app.run(debug=True)
    