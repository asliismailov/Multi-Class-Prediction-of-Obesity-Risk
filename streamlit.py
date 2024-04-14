import streamlit as st
import pandas as pd
import joblib

# Modeli yükle
model = joblib.load('lgbm_model_final.pkl')

# Başlık
st.title('Obezite Seviyesi Tahmin Uygulaması')

# Kullanıcı girişi için form
with st.form("my_form"):
    st.write("Lütfen bilgilerinizi girin:")
    
    # Özelliklerin alındığı giriş alanları
    age = st.number_input('Yaş', min_value=0, max_value=100, value=25)
    height = st.number_input('Boy (cm)', min_value=100, max_value=250, value=170)
    weight = st.number_input('Kilo (kg)', min_value=20, max_value=200, value=70)
    daily_water_intake = st.number_input('Günlük Su Tüketimi (litre)', min_value=0.0, max_value=10.0, value=2.0)
    gender = st.selectbox('Cinsiyet', ('Erkek', 'Kadın'))
    
    # Formu gönderme düğmesi
    submitted = st.form_submit_button("Tahmin Yap")

# Eğer form gönderilirse
if submitted:
    # Modelin beklediği özellik sırasına ve isimlerine uygun bir DataFrame oluştur
    input_data = pd.DataFrame({
        'age': [age],
        'height': [height],
        'weight': [weight],
        'daily_water_intake': [daily_water_intake],
        'gender': [gender]
    })
    
    # Tahmin yap
    prediction = model.predict(input_data)[0]
    
    # Tahmini göster
    st.write(f'Tahmin edilen obezite seviyesi: {prediction}')

