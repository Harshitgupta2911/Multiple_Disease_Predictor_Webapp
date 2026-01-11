# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 23:26:10 2026

@author: hp
"""

import numpy as np
import pickle 
import streamlit as st
from streamlit_option_menu import  option_menu



heart_data=pickle.load(open("heart_disease.sav",'rb'))
aneamia_data=pickle.load(open("aneamia_model.sav",'rb'))
diabetes_data=pickle.load(open("diabetes_model.sav",'rb'))
Scaler=pickle.load(open("diabetes_scaler.sav",'rb'))


with st.sidebar:
       selected=option_menu("Multiple Disease Predictor",["Heart Disease Prediction",
                            "Anaemia Prediction","Diabetes Prediction"]
                     ,default_index=0)


if(selected == 'Heart Disease Prediction'):
    st.title("❤️ Heart Disease Prediction Using ML")
    col1,col2=st.columns(2)
    with col1:     
     age = st.number_input("Age", min_value=1, max_value=120)
    with col2:
     sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=0, max_value=1)
    with col1:
     CP = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3)
    with col2: 
     Trestbps = st.number_input("Resting Blood Pressure")
    with col1: 
     Chol = st.number_input("Cholesterol")
    with col2: 
     FBS = st.number_input("Fasting Blood Sugar (1 = True, 0 = False)", min_value=0, max_value=1)
    with col1: 
     Restecg = st.number_input("Rest ECG (0–2)", min_value=0, max_value=2)
    with col2: 
     Thalach = st.number_input("Maximum Heart Rate Achieved")
    with col1:
     Exang = st.number_input("Exercise Induced Angina (1 = Yes, 0 = No)", min_value=0, max_value=1)
    with col2: 
     Oldpeak = st.number_input("Oldpeak (ST depression)", format="%.2f")
    with col1:
     Slope = st.number_input("Slope (0–2)", min_value=0, max_value=2)
    with col2: 
     CA = st.number_input("CA (0–4)", min_value=0, max_value=4)
    with col1:
     Thal = st.number_input("Thal (1–3)", min_value=1, max_value=3)
    heart_diagnosis=''
    if st.button("Predict The Heart Disease"):
             input_data=np.array([age,sex,CP,Trestbps,Chol,FBS,Restecg,Thalach,Exang,Oldpeak,Slope,CA,Thal])
             input_data=input_data.reshape(1,-1)
             heart_prediction=heart_data.predict(input_data)
             if(heart_prediction[0]==0):
              heart_diagnosis= "The person does not have a heart disease"
             else:
              heart_diagnosis= "The person has heart disease"
    st.success(heart_diagnosis)
         
    
        


if(selected=='Anaemia Prediction'):
    st.title("Aneamia Prediction Using ML")  
    col1,col2=st.columns(2)
    with col1:   
     gender=st.number_input("Gender(1 for Male and 0 for female)") 
    with col2: 
     hemoglobin=st.number_input("Hemoglobin")
    with col1:
     MCH=st.number_input("MCH")
    with col2: 
     MCHC=st.number_input("MCHC")
    with col1: 
     MCV=st.number_input("MCV")
       
    Aneamia=''
    if st.button("Predict Aneamia"):
        input_data=np.array([gender,hemoglobin,MCH,MCHC,MCV])
        input_data=input_data.reshape(1,-1)
        Aneamia_prediction=aneamia_data.predict(input_data)
        if(Aneamia_prediction[0]==0):
         Aneamia= "Hurrah!! You don't have Aneamia"
        else:
         Aneamia= "You Are Suffering From Aneamia"
       
    st.success(Aneamia)
   
    
if(selected== 'Diabetes Prediction'):
    st.title("Diabetes Prediction Using ML")
    col1,col2=st.columns(2)
    with col1:
     Pregnancies=st.number_input("Pregnancies")
    with col2: 
     Glucose=st.number_input("Glucose")
    with col1: 
     BloodPressure=st.number_input("BloodPressure")
    with col2: 
     SkinThickness=st.number_input("SkinThickness")
    with col1: 
     Insulin=st.number_input("Insulin")
    with col2: 
     BMI=st.number_input("BMI")
    with col1: 
     DiabetesPedigreeFunction=st.number_input("Diabetes Pedigree Function")
    with col2: 
     Age=st.number_input("Age")
        
    diabetes=''
    if st.button("Predict Diabetes"):
            input_data=np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
            input_data_as_numpy_array=np.asarray(input_data)
            input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
            std_data=Scaler.transform(input_data_reshape)
            prediction=diabetes_data.predict(std_data)
            if(prediction[0]==0):
              diabetes= 'You are not diabetic'
            else:
              diabetes= 'You are  diabetic'
    st.success(diabetes)
   
    
    
    
    
    
    
    
    
    
    
    