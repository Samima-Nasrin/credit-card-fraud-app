# ===========Importing the dependencies ==============
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a pandas DataFrame
credit_card_data=pd.read_csv('/creditcard.csv')

# first 5 rows of dataset
credit_card_data.head()

credit_card_data.tail()

# dataset information
credit_card_data.info()

# checking number of missing values in each column
credit_card_data.isnull().sum()

# distribution of legit and fraudulent transaction
credit_card_data['Class'].value_counts()

# 0 -> Normal transaction
# 1 -> Fraudulent transaction

# seperating data for analysis
legit=credit_card_data[credit_card_data.Class == 0]
fraud=credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical measures of the data
legit.Amount.describe()

fraud.Amount.describe()

# compare the values for both transaction
credit_card_data.groupby('Class').mean()

# ==============================Under-Sampling ================================

#build a sample dataset containing similar distribution of normal transaction and fradulent transaction
legit_sample=legit.sample(n=492)

#concatenating two dataframes
new_dataset=pd.concat([legit_sample, fraud], axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

# ============Splitting data into Features & Targets================= 

X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

X

Y

# =================Split data into Training & Test data================= 

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

# ======================Model Training================================= 

#Logistic Regression (Binary classification)
model=LogisticRegression()

# Training the logistic regression model with training data
model = LogisticRegression(max_iter=6700)
model.fit(X_train, Y_train)

# =========================Model Evaluation================================== 

#Accuracy score on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print('Accuracy on training data : ', training_data_accuracy)

#Accuracy on testing data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)

print('Accuracy on testing data : ',test_data_accuracy)

# =================================Saving the trained model===============================

import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))

# loading the save model
loaded_model=pickle.load(open('trained_model.sav', 'rb'))


def predict_transaction(input_values):
    # input_values must match feature count
    input_df = pd.DataFrame([input_values], columns=X.columns)
    prediction = loaded_model.predict(input_df)[0]

    if prediction == 0:
        return "Transaction is Legit"
    else:
        return "Transaction is Fraudulent"

# Example: simulate user input
# (This example just uses first test row)
sample_values = X_test.iloc[5].tolist()

result = predict_transaction(sample_values)
print(result)