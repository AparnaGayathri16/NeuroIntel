from flask.helpers import url_for
from flask import Flask, render_template, request, flash, redirect, send_file, send_from_directory
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os


app = Flask(__name__)

l1 = ['acidity', 'vomiting', 'indigestion', 'headache', 'blurred_and_distorted_vision',
      'excessive_hunger', 'stiff_neck', 'weakness_of_one_body_side',
      'depression', 'irritability', 'altered_sensorium', 'visual_disturbances',
      'Paralysis', 'speaking_issue', 'memory_loss', 'confusion', 'mood_swings',
      'muscle_weakness', 'hearing_problem', 'sleeplesness', 'leakage_of_urine']

disease = ['Migraine', 'BrainStroke', 'Alzheimer', 'BrainTumor', 'Parkinsons']

def HybridModel(psymptoms):
    base_estimators = [
        ('decision_tree', DecisionTreeClassifier()),
        ('random_forest', RandomForestClassifier())
    ]
    clf = StackingClassifier(estimators=base_estimators, final_estimator=RandomForestClassifier())

    df = pd.read_csv("Training.csv")
    df.replace({'prognosis': {'Migraine': 0, 'BrainStroke': 1, 'Alzheimer': 2, 'BrainTumor': 3,
                              'Parkinsons': 4}}, inplace=True)

    X = df[l1]
    y = df[["prognosis"]]
    np.ravel(y)

    clf.fit(X, np.ravel(y))

    tr = pd.read_csv("Testing.csv")
    tr.replace({'prognosis': {'Migraine': 0, 'BrainStroke': 1, 'Alzheimer': 2, 'BrainTumor': 3,
                              'Parkinsons': 4}}, inplace=True)

    X_test = tr[l1]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)

    psymptoms = psymptoms.split(',')
    l2 = [0] * len(l1)

    for k in range(0, len(l1)):
        for z in psymptoms:
            if z == l1[k]:
                l2[k] = 1

    inputtest = [l2]
    predict = clf.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if predicted == a:
            h = 'yes'
            break

    if h == 'yes':
        return f"Sorry! You are predicted to have {disease[a]}"
    else:
        return "Not Found"



model = pickle.load(open('models/model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', l1=l1)

@app.route('/predict1', methods=['POST'])
def predict():
    symptoms = request.form.get('Symptom1') + ',' + request.form.get('Symptom2') + ',' + request.form.get('Symptom3') + ',' + request.form.get('Symptom4') + ',' + request.form.get('Symptom5')
    result = HybridModel(symptoms)
    return render_template('result.html', result=result)

@app.route("/brainstroke", methods=['GET', 'POST'])
def brainstrokePage():
    return render_template('brainstroke.html')


@app.route("/alzheimer", methods=['GET', 'POST'])
def alzheimerPage():
    return render_template('alzheimer.html')

@app.route("/braintumor", methods=['GET', 'POST'])
def braintumorPage():
    
    return render_template('braintumor.html')


@app.route("/result", methods=['GET', 'POST'])
def predictPage():
    if request.method == 'POST':
        gender_Male = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension_1 = int(request.form['hypertension'])
        heart_disease_1 = int(request.form['disease'])
        ever_married_Yes = int(request.form['married'])
        work = int(request.form['work'])
        Residence_type_Urban =int( request.form['residence'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking = int(request.form['smoking'])
        work_type_Never_worked=0
        work_type_Private=0
        work_type_Self_employed=0
        work_type_children=0
        if(work==1):
           work_type_Never_worked=1
        elif work==2: 
            work_type_Private=1
        elif work==3:
            work_type_Self_employed=1
        elif work==4:
            work_type_children=1
        smoking_status_formerly_smoked=0
        smoking_status_never_smoked =0
        smoking_status_smokes=0
        if smoking==1:
            smoking_status_formerly_smoked=1
        elif smoking==2:
            smoking_status_never_smoked=1
        elif smoking==3:
            smoking_status_smokes=1
            

        input_features = [age, avg_glucose_level, bmi, gender_Male, hypertension_1, heart_disease_1, ever_married_Yes,
                          work_type_Never_worked, work_type_Private, work_type_Self_employed, work_type_children,
                          Residence_type_Urban, smoking_status_formerly_smoked, smoking_status_never_smoked,
                          smoking_status_smokes]

        features_value = [np.array(input_features)]
        features_name = ['age', 'avg_glucose_level', 'bmi', 'gender_Male', 'hypertension_1', 'heart_disease_1',
                          'ever_married_Yes', 'work_type_Never_worked', 'work_type_Private',
                          'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
                          'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

        df = pd.DataFrame(features_value, columns=features_name)
        print(df)
        prediction = model.predict(df)[0]
        print(prediction)
        if prediction == 1:
            return render_template('brainstroke.html', prediction_text='Patient has stroke risk')
        else:
            return render_template('brainstroke.html', prediction_text='Congratulations, patient does not have stroke risk')
    else:
        return render_template('brainstroke.html')
    

model1 = load_model(r"models/alz.h5", compile=False)
@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model1.predict(x), axis=1)
        alzheimerPage = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        text = "The MRI is predicted as " + str(alzheimerPage[pred[0]])
        return text
    else:
        return render_template('alzheimer.html')
    
model2 = load_model(r"models/tum.h5", compile=False)
@app.route("/pred", methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads1', f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(180, 180))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # Assuming model2 is a model trained for brain tumor prediction
        pred = np.argmax(model2.predict(x), axis=1)
        # List of class labels for brain tumor prediction
        braintumor_labels = ['GliomaTumor', 'MeningiomaTumor', 'NoTumor', 'PituitaryTumor']
        # Get the predicted class label
        predicted_label = braintumor_labels[pred[0]]
        text = "The MRI is predicted as " + str(predicted_label)
        return text
    else:
        return render_template('braintumor.html')


   
if __name__ == '__main__':
    app.run(debug=True)
 