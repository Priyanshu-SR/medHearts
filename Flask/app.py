import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os


# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(base_dir, 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details.html')
def details():
    return render_template('details.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Extracting the form data from the request
    Gender = float(request.form["Gender"])
    Age = float(request.form["Age"])
    Patient = float(request.form["Patient"])
    Severity = float(request.form["Severity"])
    BreathShortness = float(request.form["BreathShortness"])
    VisualChange = float(request.form["VisualChanges"])
    NoseBleeding = float(request.form["NoseBleeding"])
    Whendiagnosed = float(request.form["Whendiagnosed"])
    Systolic = float(request.form["Systolic"])
    Diastolic = float(request.form["Diastolic"])
    ControlledDiet = float(request.form["ControlledDiet"])

    # Creating a numpy array of the input features
    features_values = np.array([[Gender, Age, Patient, Severity, BreathShortness, VisualChange,
                                 NoseBleeding, Whendiagnosed, Systolic, Diastolic, ControlledDiet]])

    # Creating a DataFrame from the numpy array
    df = pd.DataFrame(features_values, columns=['Gender', 'Age', 'Patient', 'Severity', 'BreathShortness', 'VisualChanges',
                                                'NoseBleeding', 'Whendiagnosed', 'Systolic', 'Diastolic', 'ControlledDiet'])

    # Making the prediction using the loaded model
    prediction = model.predict(df)
    print(prediction[0])

    # Interpreting the prediction result
    if prediction[0] == 0:
        result = "HYPERTENSION (Stage-1)"
        image_file = "stage1.png"
    elif prediction[0] == 1:
        result = "HYPERTENSION (Stage-2)"
        image_file = "stage2.png"
    elif prediction[0] == 2:
        result = "HYPERTENSIVE CRISIS"
        image_file = "crisis.png"
    else:
        result = "NORMAL"
        image_file = "normal.png"

    print(result)

    # Preparing the response
    # text = "Your Blood Pressure stage is: " + result
    return render_template('result.html', result=result, image_file=image_file)


if __name__ == "__main__":
    app.run(debug=True)