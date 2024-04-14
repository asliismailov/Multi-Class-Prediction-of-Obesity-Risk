import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Modeli yükle
model = joblib.load('lgbm_model_final.pkl')

# Başlık
st.title('Obezite Seviyesi Tahmin Uygulaması')

# Kullanıcı girişi için form
with st.form("my_form"):
    st.write("Lütfen bilgilerinizi girin:")
    
# Kullanıcıdan bilgileri al
        selected_age = st.number_input("Yaş", min_value=0, max_value=150, value=30, step=1)
        selected_gender = st.radio("Cinsiyet", ["Erkek", "Kadın"])
        selected_weight = st.number_input("Kilo (kg)", min_value=20, max_value=500, value=70, step=1)
        selected_height = st.number_input("Boy (cm)", min_value=50, max_value=300, value=170, step=1)
        selected_CH2O = st.number_input("Günlük Su Tüketimi (ml)", min_value=0, max_value=10000, value=2000, step=100)
        selected_FCVC = st.number_input("Sebze Tüketilen Öğün Sayısı", min_value=0, max_value=3, value=1, step=1)

    
    # Formu gönderme düğmesi
    submitted = st.form_submit_button("Tahmin Yap")

# Eğer form gönderilirse
if submitted:
    # Modelin beklediği özellik sırasına göre bir DataFrame oluştur
    # Özellik isimleri modelin eğitildiği veri setiyle aynı olmalıdır.
    feature_values = [age, height, weight]  # Bu liste modelin beklediği özelliklerle doldurulmalıdır
    feature_names = ['age', 'height', 'weight']  # Bu da özellik isimleriyle doldurulmalıdır
    input_data = pd.DataFrame([feature_values], columns=feature_names)
    
    # Tahmin yap
    prediction = model.predict(input_data)[0]
    
    # Tahmini göster
    st.write(f'Tahmin edilen obezite seviyesi: {prediction}')
    
