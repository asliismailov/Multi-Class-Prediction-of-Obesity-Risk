import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modelin yüklenmesi
model = joblib.load('lgbm_model.pkl')

def display_about():
    st.title('Hakkımızda')
    # Hakkımızda bilgileri ve takım üyeleri detayları

# Kullanıcı girişlerini al
st.title('Obezite Durumu Tahmini')

# Modelin en çok etkilendiği özellikler
weight = st.number_input('Kilo (kg):', min_value=0.0, format="%.2f")
age = st.number_input('Yaş:', min_value=0, max_value=120)
height = st.number_input('Boy (metre):', min_value=0.1, format="%.2f")  # Metreye çevir, minimum değer 0.1 olarak güncellendi

faf = st.slider('Fiziksel aktivite frekansı (haftada kaç gün):', 0, 7, 1)
ch2o = st.slider('Günlük su tüketimi (Litre):', 0.0, 5.0, 1.0)
tue = st.slider('Teknoloji kullanım süresi (günde kaç saat):', 0, 24, 3)
fcvc = st.slider('Günde kaç kez sebze tüketiyorsunuz?', 0, 5, 2)
ncp = st.slider('Günlük ana öğün sayısı:', 1, 5, 3)
gender = st.selectbox('Cinsiyet:', ['Kadın', 'Erkek'])
family_history_with_overweight = st.selectbox('Ailede aşırı kilolu birey var mı?', ['Evet', 'Hayır'])

# BMI hesaplama ve 0 bölme hatası kontrolü
if height > 0:
    bmi = weight / (height ** 2)
else:
    bmi = 0  # height 0 ise BMI 0 olarak atanabilir veya kullanıcıya hata mesajı gösterilebilir.

# Varsayılan ve kullanıcı girdilerinin birleştirilmesi
default_values = {
    "CALC_Always": 0,
    "MTRANS_Walking": 0,
    # Diğer varsayılan değerler
    "BMI": bmi
}

user_input = {
    'Weight': weight,
    'Age': age,
    'Height': height,
    'FAF': faf,
    'CH2O': ch2o,
    'TUE': tue,
    'FCVC': fcvc,
    'NCP': ncp,
    'Gender': 1 if gender == 'Erkek' else 0,
    'family_history_with_overweight': 1 if family_history_with_overweight == 'Evet' else 0,
}

full_data = {**default_values, **user_input}

# Veri çerçevesi oluşturuluyor ve tahmin yapılıyor
input_df = pd.DataFrame([full_data])
if st.button('Tahmin Yap'):
    prediction = model.predict(input_df)
    st.write(f'Tahmin edilen Obezite Durumu: {prediction[0]}')
