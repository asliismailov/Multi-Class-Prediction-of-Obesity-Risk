import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modeli joblib ile yükle
model = joblib.load("files/lightgbm_model.pkl")

# Veri setini yükle
data = pd.read_csv("files/predicted_obesity_levels.csv")

# Kullanıcı girişlerini al
st.title('Obezite Durumu Tahmini')

# Modelin en çok etkilendiği özellikler
weight = st.number_input('Kilo (kg):', min_value=0.0, format="%.2f", value=70.0)
age = st.number_input('Yaş:', min_value=0, max_value=120, value=25)
height = st.number_input('Boy (metre):', min_value=0.1, format="%.2f", value=1.75)

faf = st.slider('Fiziksel aktivite frekansı (haftada kaç gün):', 0, 7, 1)
ch2o = st.slider('Günlük su tüketimi (Litre):', 0.0, 5.0, 1.0)
tue = st.slider('Teknoloji kullanım süresi (günde kaç saat):', 0, 24, 3)
fcvc = st.slider('Günde kaç kez sebze tüketiyorsunuz?', 0, 5, 2)
ncp = st.slider('Günlük ana öğün sayısı:', 1, 5, 3)
gender = st.selectbox('Cinsiyet:', ['Kadın', 'Erkek'])
family_history_with_overweight = st.selectbox('Ailede aşırı kilolu birey var mı?', ['Evet', 'Hayır'])

# BMI hesaplama
bmi = weight / (height ** 2)

# Veri çerçevesi oluşturuluyor
input_df = pd.DataFrame({
    'Weight': [weight],
    'Age': [age],
    'Height': [height],
    'FAF': [faf],
    'CH2O': [ch2o],
    'TUE': [tue],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'Gender': [1 if gender == 'Erkek' else 0],
    'family_history_with_overweight': [1 if family_history_with_overweight == 'Evet' else 0],
    'BMI': [bmi]
})

# Tahmin yapılıyor
if st.button('Tahmin Yap'):
    prediction = model.predict(input_df)
    st.write(f'Tahmin edilen Obezite Durumu: {prediction[0]}')
