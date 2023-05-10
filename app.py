import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pycaret.classification import load_model 

st.header('"This is a machine that predicts the behavior of bank customers" ')
st.sidebar.header('Input Parameteres')
def user_input_features():
    CreditScore=st.sidebar.slider('CreditScore',350,850,650)
    Geography=st.sidebar.selectbox('Geography',['France','Germany','Spain'])
    Gender=st.sidebar.selectbox('Gender',['Male','Female'])
    Age=st.sidebar.slider('Age',18,85,41)
    Tenure=st.sidebar.slider('Tenure',0,10,5)
    Balance=st.sidebar.slider('Balance',0.0,250898.090000,81330.251765)
    NumOfProducts=st.sidebar.slider('NumOfProducts',1.0,4.0,1.5)
    HasCrCard=st.sidebar.radio('HasCrCard',[0,1])
    IsActiveMember=st.sidebar.radio('IsActiveMember',[0,1])
    EstimatedSalary=st.sidebar.slider('EstimatedSalary',11.5,200000.0,100000.0)
    data={'CreditScore':CreditScore,
          'Geography':Geography,
          'Gender':Gender,
          'Age':Age,
          'Tenure':Tenure,
          'Balance':Balance,
          'NumOfProducts':NumOfProducts,
          'HasCrCard':HasCrCard,
          'IsActiveMember':IsActiveMember,
          'EstimatedSalary':EstimatedSalary}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()
st.subheader('User Input Parametres')
st.write(df)
st.subheader('Name Of Targets')

dataset=pd.read_csv('Churn.csv')
labels_name=dataset['Exited'].unique()
label_fram=pd.DataFrame(labels_name,index=['Cat1','Cat2'],columns=['Names'])
st.write(label_fram)
mymodel=load_model('mymodel')
prediction=mymodel.predict(df)
st.subheader('Prediction is:')
st.write(str(prediction))
