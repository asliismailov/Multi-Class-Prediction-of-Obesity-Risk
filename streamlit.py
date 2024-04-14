import streamlit as st
import pandas as pd
import joblib

# Streamlit sayfa konfigürasyonu
st.set_page_config(layout="wide", page_title="Obezite Riskinin Çok Sınıflı Tahmini", page_icon="🎈")

# Modeli yükleyecek fonksiyon
@st.cache(allow_output_mutation=True)
def get_model():
    return joblib.load('lgbm_model_final.pkl')

# Model yükleniyor
model = get_model()

# Cinsiyetler için sayısal değerleri içeren sözlük
gender_dict = {'Male': 1, 'Female': 0}

# BMI hesaplama fonksiyonu
def calculate_bmi(weight, height):
    height_m = height / 100  # boyu cm'den metreye çevir
    bmi = weight / (height_m ** 2)
    return bmi

# Tahmin fonksiyonu
def predict_obesity_risk(age, gender, weight, height, ch2o, bmi):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight': [weight],
        'Height': [height],
        'CH2O': [ch2o],
        'BMI': [bmi]
    })
    
    # Sütunları modelin eğitildiği sıraya göre düzenle
    expected_features = ['Age', 'Gender', 'Weight', 'Height', 'CH2O', 'BMI']  # ve diğer tüm özellikler
    prediction = model.predict(input_data[expected_features])
    return prediction

# Ana Sayfa layout
main_tab, chart_tab, prediction_tab = st.columns(3)

# Ana Sayfa içeriği
with main_tab:
    st.header("Proje Hakkında")
    # Proje hakkındaki açıklamalarınız buraya eklenebilir.

# Grafikler sekmesi
with chart_tab:
    st.header("Analitik Grafikler")
    # Grafikleriniz buraya eklenebilir.

# Tahmin sekmesi
with prediction_tab:
    st.header("Model ile Tahmin Yapma")

def predict_obesity_risk(age, gender, weight, height, ch2o, bmi):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight': [weight],
        'Height': [height],
        'CH2O': [ch2o],
        'BMI': [bmi]
    })

    # Modelin beklendiği sıraya göre sütunları yeniden düzenle
    expected_features = ['Age', 'Gender', 'Weight', 'Height', 'CH2O', 'BMI']
    input_data = input_data[expected_features]

    prediction = model.predict(input_data)
    return prediction

    
    with st.form(key='obesity_form'):
        selected_age = st.number_input("Yaş", min_value=0.0, max_value=150.0, value=30.0, step=1.0, format="%.2f")
        selected_gender = st.selectbox("Cinsiyet", ('Male', 'Female'))
        selected_weight = st.number_input("Kilo (kg)", min_value=20.0, max_value=500.0, value=70.0, step=0.1, format="%.2f")
        selected_height = st.number_input("Boy (cm)", min_value=50.0, max_value=300.0, value=170.0, step=0.1, format="%.2f")
        selected_ch2o = st.number_input("Günlük Su Tüketimi (ml)", min_value=0.0, max_value=10000.0, value=2000.0, step=0.1, format="%.2f")
        submit_button = st.form_submit_button(label='Tahminle')

        if submit_button:
            gender_numeric = gender_dict[selected_gender]
            bmi_value = calculate_bmi(selected_weight, selected_height)
            prediction = predict_obesity_risk(selected_age, gender_numeric, selected_weight, selected_height, selected_ch2o, bmi_value)
            st.write("Tahmin Edilen Obezite Riski:", prediction[0])  # prediction bir dizi olduğundan, ilk öğeyi alıyoruz
