# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:42:06 2023

@author: HP
"""

import streamlit as st
import pickle
import pandas as pd

# Load the trained model from the pickle file
with open('C:/Users/user/Desktop/Stuff/Project/Clustering/cluster_km.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to get input features from user
def input_features():
    
    Birth_Rate = st.sidebar.number_input('Enter the Birth Rate value')
    CO2_Emissions = st.sidebar.number_input('Enter the Carbon Emission value')
    Energy_Usage = st.sidebar.number_input('Enter the Energy Usage Value')
    GDP = st.sidebar.number_input('Enter the GDP value')
    Health_Exp_Capita = st.sidebar.number_input('Enter the Health Expenditure Per Capita Value')
    IMR = st.sidebar.number_input("Enter the Infant Mortality Rate value (0-1)",min_value=0.0,max_value=1.0,step=0.001)
    InternetUsage = st.sidebar.number_input("Enter the Internet Usage Value (0-1)",min_value=0.0,max_value=1.0,step=0.001)
    LifeExpF = st.sidebar.number_input('Enter the Life Expectancy of Female')
    LifeExpM = st.sidebar.number_input('Enter the Life Expectancy of Male')
    Mobile = st.sidebar.number_input("Enter the Mobile Phone Usage Value (0-1)",min_value=0.0,max_value=1.0,step=0.01)
    Pop_1 = st.sidebar.number_input("Enter the Value of % Population of Ages between 0-14",min_value=0.0,max_value=1.0,step=0.01)
    Pop_2 = st.sidebar.number_input("Enter the value of % Population of Ages between 15-64",min_value=0.0,max_value=1.0,step=0.01)
    Pop_3 = st.sidebar.number_input("Enter the value of % Population of Age above 65",min_value=0.0,max_value=1.0,step=0.01)
    Population = st.sidebar.number_input("Enter the Total Population Count")
    Urban = st.sidebar.number_input("Enter the Value of % Population in Urban")
    T_in = st.sidebar.number_input("Enter the Tourism Inbound Value")
    T_out = st.sidebar.number_input("Enter the Tourism Outbound Value")
    data = {
            'Birth Rate':Birth_Rate,
            'CO2 Emissions':CO2_Emissions,
            'Energy Usage':Energy_Usage,
            'GDP':GDP,
            'Health Exp/Capita':Health_Exp_Capita,
            'Infant Mortality Rate':IMR,
            'Internet Usage':InternetUsage,
            'Life Expectancy Female':LifeExpF,
            'Life Expectancy Male':LifeExpM,
            'Mobile Phone Usage':Mobile,
            'Population 0-14':Pop_1,
            'Population 15-64':Pop_2,
            'Population 65+':Pop_3,
            'Population Total':Population,
            'Population Urban':Urban,
            'Tourism Inbound':T_in,
            'Tourism Outbound':T_out}
    features = pd.DataFrame(data,index=[0])
    return features

# Define a function to predict the cluster and output result
def predict_cluster(features):
    prediction = model.predict(features)[0]
    output = analysis(prediction)
    return output

# Define a function to analyze the prediction result
def analysis(prediction):
    if prediction == 0:
        return 'Less Developed country'
    elif prediction == 1:
        return 'More Developed Country'
    elif prediction == 2:
        return 'Most developed Country'
    else:
        return 'Least developed Country'

# Define the Streamlit app
def main():
    # Set the app title
    st.title('Country Cluster Prediction')

    # Set the app sidebar
    st.sidebar.title('Input Features')
    features = input_features()

    # Display the input features
    st.write('Input Features:')
    st.write(features)

    # Make prediction and display the result
    if st.sidebar.button('Predict'):
        result = predict_cluster(features)
        st.write('Prediction:')
        st.write(result)

# Run the app
if __name__ == '__main__':
    main()
