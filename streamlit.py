import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the model with joblib
model = joblib.load("lgbm_model.pkl")

# Load the dataset
data = pd.read_csv("predicted_obesity_levels.csv")

def display_about():
    st.title('About Us')
    st.write('''
    This application is developed to predict obesity status based on users' health data.
    The app analyzes various nutrition and lifestyle data to help assess the risk of obesity.
    Our project aims to provide informative content to help users better understand and manage their health.
    ''')

    st.subheader('Team Members')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('asli.jpeg', width=150)
        st.markdown('**Aslı Öztürk**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/ozturk-asli/)')
        st.write('Aslı is an expert researcher in data science and machine learning. She has worked with big data on various projects and gained experience in health technologies.')

    with col2:
        st.image('begum.jpeg', width=150)
        st.markdown('**Begüm Baybora**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/begumbaybora/)')
        st.write('Begüm is a software engineer with extensive experience in artificial intelligence and data analysis. She has developed solutions in educational technologies and the healthcare sector.')

    with col3:
        st.image('ugur.jpeg', width=150)
        st.markdown('**Uğur Can Odabaşı**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/ugurcanodabasi)')
        st.write('Uğur Can has in-depth knowledge in software development and systems engineering. He has provided technology solutions across various industries and has led technical projects.')

# Homepage
def home_page():
    st.title('Home Page')

    # World obesity map as a placeholder static image
    st.image('Obesity_rate_(WHO,_2022).png', caption='World Obesity Map')

    # Text about obesity being a modern epidemic
    st.write("""
    ## Obesity: The Epidemic of Our Age

    Obesity affects millions of people worldwide and is a rapidly spreading serious health issue.
    Individuals with a body mass index (BMI) over 30 are classified as obese, which significantly increases the risk of various chronic diseases.
    Obesity can lead to health issues such as heart disease, type 2 diabetes, certain types of cancer, and musculoskeletal disorders.

    Combating obesity requires not only individual effort but also a global effort.
    Adopting healthy eating habits, engaging in regular physical activity, and making healthy lifestyle choices are key in fighting obesity.
    Communities and governments should develop and implement policies to facilitate access to healthy foods, promote physical activity, and increase health education.
    """)

# Dynamic Graphs
def dynamic_graphs():
    col1, col2 = st.columns(2)

    with col1:
       st.header("Correlation Matrix")
       st.image("correlation.png")

    with col2:
       st.header("Shap")
       st.image("SHAP.png")

# Obesity Prediction
def predict_obesity():
    st.title('Predict Obesity Status')

    # Collect user inputs
    gender = st.selectbox('Your Gender', ['Male', 'Female'])
    age = st.number_input('Your Age', min_value=1, max_value=100, value=25)
    height = st.number_input('Your Height (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Your Weight (kg)', min_value=20, max_value=200, value=70)
    family_history = st.selectbox('Family history of obesity?', ['Yes', 'No'])
    favc = st.selectbox('Frequency of high caloric food consumption?', ['Never', 'Occasionally', 'Often'])
    fcvc = st.selectbox('Frequency of vegetable consumption?', ['Never', 'Occasionally', 'Often', 'Very Often'])
    ncp = st.selectbox('Number of main meals per day?', ['1', '2', '3', '4 or more'])
    caec = st.selectbox('Eating behavior?', ['Never', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox('Do you smoke?', ['Yes', 'No'])
    ch2o = st.slider('Daily water consumption (liters)', 0.0
