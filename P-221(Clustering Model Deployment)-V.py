# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:16:34 2023

@author: 91944
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

with open('trained_model_clustering.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = model['scaler']
pca = model['pca']
kmean = model['kmean']

def Cluster_prediction (input_data):

    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
    input_scaled = scaler.transform(input_data_reshaped)
    
    input_pca = pca.transform(input_scaled)
        
    prediction =kmean.predict(input_pca)
    
    
    print (prediction)

    if (prediction[0] == 0):
        return 'Developing Country' 
    elif (prediction[0] == 1):
        return ' highly developed countries-socio-economically forward countries'
    elif (prediction[0] == 2):
        return ' socio-economically backward countries/under developed countries'
    elif (prediction[0] == 3):
        return 'Moderately Developed'
    
def main():

    # giving a title

    st.title('World Developement Clusters')

    # getting the input data from the user
    Country= st.text_input("Country Name:")

    Birth_Rate = st.slider ('Birth Rate', 0.001, 0.06, 0.001)

    CO2_Emissions = st.text_input('C02 Emissions')
    
    GDP = st.text_input('Total GDP')

    Health_Exp_Capita = st.slider('Health Exp/Capita', 2, 9988, 100)
   
    Health_Exp_GDP = st.slider('Health Exp % GDP', 0.02, 0.9, 0.45)
    
    Infant_Mortality_Rate = st.slider('Infant Mortality Rate', 0.002, 0.141, 0.05)

    Internet_Usage = st.slider('Internet Usage', 0.0, 1.0, 0.1)

    Life_Expectancy_Female = st.slider('Life Expectancy Female', 1, 99, 70)
    
    Life_Expectancy_male = st.slider('Life Expectancy Male', 1, 99, 70)

    Mobile_Phone_Usage = st.slider('Mobile Phone Usage', 0.0, 3.0, 0.5) 

    Population_8_14 = st.slider ('Population 0-14 %', 0.1, 0.5, 0.05)
    Population_15_64 = st.slider ('Population 15-64 %', 0.2, 0.8, 0.05)
    Population_65above = st.slider ('Population 65% +', 0.001, 0.4, 0.05)

    Population_Total = st.text_input('Total population')

    Population_Urban = st.slider('Population Urban %', 0.05, 1.0, 0.01)
    
    
    Tourism_Inbound = st.text_input('Tourism inbound')
    
    Tourism_outbound = st.text_input('Tourism outbound')
    
    Business_Tax_Rate = st.slider('Business Tax Rate',0,100,46)
    
    Days_to_Start_Business = st.slider('Days to Start Business', 1, 700, 40)
    
    Ease_of_Business = st.slider('Ease of Business',10,200,15)
    
    Energy_Usage = st.text_input('Energy Usage')
    
    Hours_todo_Tax= st.slider('Hours to do Tax',100,990,270)
    
    Lending_Interest = st.text_input('Lending Interest',0.005,5,0.01)    
    

   #code for Prediction

    Predict = ''

    #creating a button for Prediction

    if st.button("Submit"):
        Predict = Cluster_prediction ([Birth_Rate,CO2_Emissions,GDP,Health_Exp_Capita,Health_Exp_GDP,
                                       Infant_Mortality_Rate,Internet_Usage,Life_Expectancy_Female,Life_Expectancy_male,
                                       Mobile_Phone_Usage,Population_8_14,Population_15_64,Population_65above,
                                       Population_Total,Population_Urban,Tourism_Inbound,Tourism_outbound,Business_Tax_Rate,
                                       Days_to_Start_Business,Ease_of_Business,Energy_Usage,Hours_todo_Tax,Lending_Interest])

    st.success(Predict)
    

aif __name__ ==  '__main__':
    main()   
