import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import ADASYN
from sklearn.metrics import *
from xgboost import XGBClassifier

#Main header file of the project
st.header("Credit Card Approval Prediction")

#Read name of the user
st.text_input("Enter your Name: ", key="name")

#Load the final pre-processed dataset on which the models will be trained
#date pre-processing and cleaning is done in phase-1 with original data
data = pd.read_csv("trav.csv")

# Preview of the datset
if st.checkbox('Show Training Dataframe'):
    data

#Read the appliaction details that has been given to the GUI by any random user
st.subheader("Please provide details of your application!")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency = st.radio(
        'Name of the Agency',
        np.unique(data['Agency'])) 
 
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency_Type = st.radio(
        'Name of the Agency Type',
        np.unique(data['Agency_Type']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Dist_Channel = st.radio(
        'Name of the Dist Channel',
        np.unique(data['Dist_Channel']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Prod_Name = st.radio(
        'Name of the Product',
        np.unique(data['Prod_Name']))

left_column, right_column = st.columns(2)
with left_column:
    inp_Destination = st.radio(
        'Name of the Destination',
        np.unique(data['Destination']))  

input_Net_Sales = st.slider('Enter Net Sales', 0.0, max(data["Net_Sales"]), 100.0)
input_Commission = st.slider('Enter Commission Value', 0.0, max(data["Commission"]), 100.0)
input_Age = st.slider('Enter Age', 0, max(data["Age"]), 100)

uploaded_file = st.file_uploader("Upload a dataframe(CSV) similar to the above training dataframe") 
if uploaded_file is not None:   
    my_data2 = pd.read_csv(uploaded_file)   
    my_data2

# predict wether the applicant will default or not not if the credit card is issued
if st.button('Make Prediction'):
    inputs = np.expand_dims([inp_Agency, inp_Agency_Type, inp_Dist_Channel, inp_Prod_Name, input_Commission, input_Net_Sales, input_Age],0)
    #Training the best model(XGBoost) 
    if uploaded_file is not None:
        X = my_data2.drop(['Claim'], axis=1)
        y = my_data2['Claim']
    else:    
        X = data.drop(['Claim'], axis=1)
        y = data['Claim']
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    #adasyn = ADASYN()
    #X_train,y_train = adasyn.fit_resample(X_train,y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_xgboost_model = XGBClassifier()
    best_xgboost_model.fit(X_train, y_train)
    #Make prediction and print output
    prediction = best_xgboost_model.predict(inputs)
    if prediction:
        st.error("Sorry, Your Credit Card will be Declined as you may default in future")
    else:
        st.success("Congratulations, Your Credit Card will be Approved")

    st.write(f"Thank you {st.session_state.name}! We hope you liked our project.")
