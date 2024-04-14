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
    
    # Özelliklerin alındığı giriş alanları, modelin beklediği özelliklere göre ayarlanmalıdır.
    # Bu sadece bir örnektir, gerçek modelinize göre düzenlenmelidir.
    age = st.number_input('Yaş', min_value=0, max_value=100, value=25)
    height = st.number_input('Boy (cm)', min_value=100, max_value=250, value=170)
    weight = st.number_input('Kilo (kg)', min_value=20, max_value=200, value=70)
    # Buraya modelin gerektirdiği diğer özelliklerin girişleri eklenecek
    
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
