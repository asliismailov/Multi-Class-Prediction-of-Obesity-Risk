import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Streamlit sayfa konfigürasyonu
st.set_page_config(layout="wide", page_title="Obezite Riskinin Çok Sınıflı Tahmini", page_icon="🎷")

# Özel önbellek ayarları
@st.cache(allow_output_mutation=True, persist=True, suppress_st_warning=True, show_spinner=False)
def get_data():
    dataframe = pd.read_csv("predicted_obesity_levels.csv")
    return dataframe

@st.cache
def get_model():
    model = joblib.load('lgbm_model_final.pkl')
    return model

model = get_model()

# Ana Sayfa layout
main_tab, chart_tab, prediction_tab = st.columns(3)

# Ana Sayfa içeriği
with main_tab:
    st.header("Proje Hakkında")
    st.write(
        """Bu projenin amacı, bireylerde kardiyovasküler hastalıklarla ilişkili obezite riskini tahmin etmek için çeşitli faktörleri kullanmaktır. Kardiyovasküler hastalıklar, dünya genelinde sağlık sorunlarının önde gelen nedenlerinden biri olarak kabul edilmektedir. Bu hastalıkların birçoğu obezite ile doğrudan ilişkilidir. Bu nedenle, obeziteyi öngörmek ve bu konuda farkındalık yaratmak önemlidir.""")

    st.subheader("Veri Seti ve Hedef")
    st.write(
        """Bu projede kullanılan veri seti, bireylerin demografik bilgilerini, yaşam tarzı alışkanlıklarını ve fizyolojik ölçümlerini içerir. Ölçümler arasında boy, kilo, günlük su tüketimi, fiziksel aktivite düzeyi gibi faktörler bulunmaktadır. Veri setindeki her bir satır, bir bireyi temsil eder ve bu bireylerin obezite durumları "NObeyesdad" sütununda belirtilmiştir.""")

with chart_tab:
    st.header("Analitik Grafikler")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Korelasyon Matrisi")
        st.image("korelasyon.png")

    with col2:
        st.subheader("SHAP Değerleri")
        st.image("SHAP.png")

# Tahmin sekmesi
with prediction_tab:
    st.header("Model ile Tahmin Yapma")
    st.subheader("Tahmin")

    # Kullanıcı girdileri
    selected_age = st.number_input("Yaş", min_value=0, max_value=150, value=30, step=1)
    
    # Cinsiyet seçimi için güncellenmiş kısım
    gender_options = {'Erkek': 'Male', 'Kadın': 'Female'}
    selected_gender = st.radio("Cinsiyet", list(gender_options.keys()))

    selected_weight = st.number_input("Kilo (kg)", min_value=20, max_value=500, value=70, step=1)
    selected_height = st.number_input("Boy (cm)", min_value=50, max_value=300, value=170, step=1)
    selected_ch2o = st.number_input("Günlük Su Tüketimi (ml)", min_value=0, max_value=10000, value=2000, step=100)

    def predict_obesity_risk(age, gender, weight, height, ch2o):
        # Modeli kullanarak tahmin yapma
        data = {
            'Age': [age],
            'Gender': [gender],
            'Weight': [weight],
            'Height': [height],
            'CH2O': [ch2o]
        }
        df = pd.DataFrame(data)
        
        # Modelin özellik sayısını kontrol etme
        if df.shape[1] != model.n_features_:
            st.error("Modelin beklediği özellik sayısı ile girdi verisi uyumlu değil.")
            return None
        
        prediction = model.predict(df)[0]
        return prediction

    if st.button("Tahminle"):
        # Cinsiyetin sayısal değerine dönüştürülmesi
        gender_numeric = gender_dict[gender_options[selected_gender]]
        
        # Tahmin fonksiyonunu çağırma
        prediction = predict_obesity_risk(selected_age, gender_numeric, selected_weight, selected_height, selected_ch2o)
        st.write("Tahmin Edilen Obezite Riski:", prediction)
