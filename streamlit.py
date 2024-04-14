import streamlit as st
import pandas as pd
import joblib

# Streamlit sayfa konfigürasyonu
st.set_page_config(layout="wide", page_title="Obezite Riskinin Çok Sınıflı Tahmini", page_icon="🎈")

# Modeli ve veriyi yükleyecek fonksiyonlar
@st.cache(allow_output_mutation=True, persist=True, suppress_st_warning=True, show_spinner=False)
def get_data():
    return pd.read_csv("predicted_obesity_levels.csv")

@st.cache(allow_output_mutation=True)
def get_model():
    return joblib.load('lgbm_model_final.pkl')

# Cinsiyetler için sayısal değerleri içeren sözlük
gender_dict = {'Male': 1, 'Female': 0}
gender_options = {'Erkek': 'Male', 'Kadın': 'Female'}

# Model ve veri yükleniyor
model = get_model()
data = get_data()

# Tahmin fonksiyonu
def predict_obesity_risk(age, gender, weight, height, ch2o):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight': [weight],
        'Height': [height],
        'CH2O': [ch2o]
    })
    
    # Özelliklerin doğru sıra ve sayıda olduğundan emin olun
    # Bu özellikler ve sıraları, modeli eğitirken kullandığınız veri setine göre olmalı
    # Tahmin yerine predict_proba kullanıyorum, çünkü olasılık tahmini yapmak daha uygun olabilir
    prediction_proba = model.predict_proba(input_data)
    return prediction_proba

# BMI hesaplama fonksiyonu
def calculate_bmi(height, weight):
    bmi = weight / (height ** 2)
    return bmi

# Ana Sayfa layout
main_tab, chart_tab, prediction_tab = st.tabs(["Ana Sayfa", "Grafikler", "Model"])

# Ana Sayfa içeriği
with main_tab:
    st.header("Proje Hakkında")
    st.write("""
        Bu projenin amacı, bireylerde kardiyovasküler hastalıklarla ilişkili obezite riskini tahmin etmek için çeşitli faktörleri kullanmaktır.
        """)
    # İsterseniz veri seti özetini ve diğer bilgileri de buraya ekleyebilirsiniz.

# Grafikler sekmesi
with chart_tab:
    st.header("Analitik Grafikler")
    # İsterseniz grafikleri burada gösterebilirsiniz.

# Tahmin sekmesi
with prediction_tab:
    st.header("Model ile Tahmin Yapma")

    with st.form(key='obesity_form'):
        selected_age = st.number_input("Yaş", min_value=0, max_value=150, value=30, step=1)
        selected_gender = st.radio("Cinsiyet", list(gender_options.keys()))
        selected_weight = st.number_input("Kilo (kg)", min_value=20, max_value=500, value=70, step=1)
        selected_height = st.number_input("Boy (cm)", min_value=50, max_value=300, value=170, step=1)
        selected_ch2o = st.number_input("Günlük Su Tüketimi (ml)", min_value=0, max_value=10000, value=2000, step=100)
        
        submit_button = st.form_submit_button(label='Tahminle')  # Submit butonu eklendi

        if submit_button:
            # Kullanıcı seçimini sayısal değere dönüştürme
            gender_numeric = gender_dict[gender_options[selected_gender]]
            # BMI hesaplama
            bmi = calculate_bmi(selected_height, selected_weight)
            # Tahmin fonksiyonunu çağırma
            prediction = predict_obesity_risk(selected_age, gender_numeric, selected_weight, selected_height, selected_ch2o)
            st.write("Tahmin Edilen Obezite Riski:", prediction)
            st.write("BMI:", bmi)
