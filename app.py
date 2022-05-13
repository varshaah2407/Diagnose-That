from flask import Flask,render_template,request
import pickle
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
app = Flask(__name__)

#Importing all the pretrained models
model1 = load_model('static/Alzheimers_ridhi2.h5');
model2 = pickle.load(open('static/diabetes.pkl','rb'))
model3 = pickle.load(open('static/heartcancer.pkl','rb'))
model4 = pickle.load(open('static/lungcancer.pkl','rb'))
model5 = pickle.load(open('static/urinarydisease.pkl','rb'))
model6 = load_model('static/tuberculosis1.h5'); 

@app.route('/')
def hello_world():
    return render_template("nav1.html")

## Inital questionnaire on intro page
@app.route('/intro',methods=['POST','GET'])
def intro():
    if request.method == 'POST':
        name = request.form['fname']
        # age = request.form['age']
        # gender = request.form['gender']

        message="Hello "+name+" Here are the disease prediction!"
        return render_template('nav1.html',message=message)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(300,300))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    print(preds)
    return preds

### Prediction and file uploading function for Alziehmer
@app.route('/predictbrain',methods=['POST','GET'])
def predictbrain():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        my_prediction = model_predict(file_path, model1)
        # pred_class = decode_predictions(my_prediction, top=1)
        # result = str(pred_class[0][0][1])   
        if(my_prediction==1):
            return render_template('alziemerresult1.html')
        else:
            return render_template('alziemerresult0.html')
    else:
        return render_template('alziemer.html')



### Prediction and file uploading function for tb 
@app.route('/predicttb',methods=['POST','GET'])
def predicttb():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        my_prediction = model_predict(file_path, model6)
        # pred_class = decode_predictions(my_prediction, top=1)
        # result = str(pred_class[0][0][1])   
        if(my_prediction==1):
            return render_template('tbResult1.html')
        else:
            return render_template('tbResult0.html')
    else:
        return render_template('tb.html')



## Predict Diabetes 
@app.route('/predictdiabetes',methods=['POST','GET'])
def predictdiabetes():
    if request.method == 'POST':

        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bp']
        st = request.form['thickness']
        insulin = request.form['insulin']
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = model2.predict(data)

        if(my_prediction==1):
            return render_template('diabetesResult1.html')
        else:
            return render_template('diabetesResult0.html')
    else:
        return render_template('diabetes.html')
        

# Prediction of Heart Disease
@app.route('/predictdisease',methods=['POST','GET'])
def predictdisease():
    if request.method == 'POST':

        age = int(request.form['age'])
        gender = int(request.form['gender'])
        chest_pain = int(request.form['chest_pain'])
        resting_bp = int(request.form['resting_bp'])
        cholestrol = int(request.form['cholestrol'])
        fasting_bs = int(request.form['fasting_bs'])
        resting_ecg = int(request.form['resting_ecg'])
        maxHR = int(request.form['maxHR'])
        exercise = int(request.form['exercise'])
        old_peak = float(request.form['old_peak'])
        st_slope = int(request.form['st_slope'])

        data = np.array([[age, gender, chest_pain, resting_bp,cholestrol,fasting_bs,resting_ecg,maxHR,exercise,old_peak,st_slope]])
        my_prediction = model3.predict(data)

        if(my_prediction==1):
            return render_template('heartResult1.html')
        else:
            return render_template('heartResult0.html')
    else:
        return render_template('heartCancer.html')
#Lung Cancer Prediction
@app.route('/predictlung',methods=['POST','GET'])
def predictlung():
    if request.method == 'POST':

        gender = int(request.form['gender'])
        age = int(request.form['age'])
        smoke = int(request.form['smoke'])
        fingers = int(request.form['fingers'])
        anxious = int(request.form['anxious'])
        pressure = int(request.form['pressure'])
        chronic = int(request.form['chronic'])
        fatigue = int(request.form['fatigue'])
        allergy = int(request.form['allergy'])
        wheeze = float(request.form['wheeze'])
        alcohol = int(request.form['alcohol'])
        cough = int(request.form['cough'])
        breathe = int(request.form['breathe'])
        swallowing = int(request.form['swallowing'])
        chest = int(request.form['chest'])

        data = np.array([[gender, age, smoke, fingers,anxious,pressure,chronic,fatigue,allergy,wheeze,alcohol,cough,breathe,swallowing,chest]])
        my_prediction = model4.predict(data)

        if(my_prediction==1):
            return render_template('lungCancerResult1.html')
        else:
            return render_template('lungCancerResult0.html')
    else:
        return render_template('lungCancer.html')


## Prediction of Urinary Disease
@app.route('/predicturinary',methods=['POST','GET'])
def predicturinary():
    if request.method == 'POST':

        temp = float(request.form['temperature'])
        nausea = int(request.form['nausea'])
        pain = int(request.form['pain'])
        urinate = int(request.form['urinate'])
        abdomen = int(request.form['abdomen'])
        burning = int(request.form['burning'])

        data = np.array([[temp,nausea, pain, urinate, abdomen, burning]])
        my_prediction = model5.predict(data)

        if(my_prediction==1):
            return render_template('urinaryresult1.html')
        else:
            return render_template('urinaryresult0.html')
    
    else:
        return render_template('urinary_disease.html')



if __name__ == "__main__":
    app.run(debug=True)