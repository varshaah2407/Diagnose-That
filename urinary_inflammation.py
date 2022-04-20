import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# dataframe 
disease = pd.read_csv  ("acute_inflammations.csv")


disease["Temperature"] = [float(str(i).replace(",", ".")) for i in disease["Temperature"]]
disease.Occurrence_of_nausea = disease.Occurrence_of_nausea.map(dict(yes=1, no=0))
disease. Lumbar_pain = disease. Lumbar_pain .map(dict(yes=1, no=0))
disease.Micnutrition_pains= disease.Micnutrition_pains.map(dict(yes=1, no=0))
disease. Urine_pushing = disease.Urine_pushing.map(dict(yes=1, no=0))
disease.Burning_of_urethra = disease.Burning_of_urethra .map(dict(yes=1, no=0))
disease.Decision_Inflammation_of_urinary_bladder = disease.Decision_Inflammation_of_urinary_bladder .map(dict(yes=1, no=0))
disease. Decision_Nephritis_or_renal_pelvis_origin = disease. Decision_Nephritis_or_renal_pelvis_origin.map(dict(yes=1, no=0))


X = disease.drop(columns = 'Decision_Inflammation_of_urinary_bladder',axis = 1)
Y = disease['Decision_Inflammation_of_urinary_bladder']

# Splitting the test and train data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y,random_state = 2)
print(X.shape,X_train.shape,X_test.shape)
model = LogisticRegression()
model.fit(X_train,Y_train)

# Model Evaluation
# Accuracy score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data:',training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data: ',testing_data_accuracy)

# Building a predictive system
input_data = (41.1,1,0,0,1,0,0)
input_data_as_numpy_array = np.asarray(input_data)
# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print("Prediction: ",prediction)
