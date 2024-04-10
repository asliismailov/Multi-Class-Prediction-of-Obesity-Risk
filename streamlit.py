import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle

# Özel bir önbellek yöneticisi tanımlama
custom_cache = st.cache(allow_output_mutation=True, persist=True, suppress_st_warning=True, show_spinner=False)

st.set_page_config(layout = "wide", page_title="Obezite Riskinin Çok Sınıflı Tahmini", page_icon="🎷")

@st.cache
def get_data():
    dataframe = pd.read_csv('predicted_obesity_levels.csv')
    return dataframe

# Modeli yükle
@st.cache
def get_pipeline():
    pipeline = joblib.load('lgbm_model_final.pkl')
    return pipeline

main_tab, chart_tab, prediction_tab = st.tabs(["Ana Sayfa", "Grafikler", "Model"])

# Ana Sayfa ########################################################

left_col, right_col = main_tab.columns(2)

left_col.write("""Bu projenin amacı, bireylerde kardiyovasküler hastalıklarla ilişkili obezite riskini tahmin etmek için çeşitli faktörleri kullanmaktır. Kardiyovasküler hastalıklar, dünya genelinde sağlık sorunlarının önde gelen nedenlerinden biri olarak kabul edilmektedir. Bu hastalıkların birçoğu obezite ile doğrudan ilişkilidir. Bu nedenle, obeziteyi öngörmek ve bu konuda farkındalık yaratmak önemlidir.""")

left_col.write("""Veri Seti ve Hedef
Bu projede kullanılan veri seti, bireylerin demografik bilgilerini, yaşam tarzı alışkanlıklarını ve fizyolojik ölçümlerini içerir. Ölçümler arasında boy, kilo, günlük su tüketimi, fiziksel aktivite düzeyi gibi faktörler bulunmaktadır. Veri setindeki her bir satır, bir bireyi temsil eder ve bu bireylerin obezite durumları "NObeyesdad" sütununda belirtilmiştir.""")

#TAVSİYE:Veri setinin bir kısmı eklenebilir

right_col.write("""Kullanılan Algoritmalar
Bu proje, LightGBM makine öğrenimi modeli kullanmaktadır. LightGBM, yüksek performanslı ve hızlı bir gradyan arttırma (gradient boosting) algoritmasıdır. Bu algoritma, veri setindeki örüntüleri öğrenerek ve karmaşık ilişkileri modelleyerek obezite riskini tahmin etmek için kullanılır.""")

right_col.write("""Uygulama: Streamlit ile Model Tahmini
Bu projede, geliştirilen modelin kullanıcı dostu bir arayüz ile sunulması amaçlanmıştır. Streamlit adlı Python kütüphanesi, basit ve etkileşimli web uygulamaları oluşturmayı sağlar. Bu projede, geliştirilen LightGBM modeli Streamlit arayüzü ile entegre edilmiştir.
Kullanıcılar, arayüz üzerinden bireysel özellikleri girebilir ve modele besleyerek obezite risk tahminini alabilirler. Bu tahminler, bireylerin normal kilolu, aşırı kilolu, obez veya aşırı obez olma riskini belirtir.""")

#TAVSİYE: IMAGE EKLENEBİLİR
#right_col.image("spoti.jpg")

# Grafikler ########################################################

import streamlit as st

col1, col2 = chart_tab.columns(2)

with col1:
   st.header("Korelasyon Matrisi")
   st.image("korelasyon.png")

with col2:
   st.header("Shap")
   st.image("SHAP.png")

#Tahmin ########################################################

if prediction_tab.button("Model"):
    model_cont = st.container()
    model_cont.subheader("Tahmin")
    col1, col2, col3, col4, col5 = model_cont.columns(5)
    
    # Kullanıcıdan bilgileri al
    selected_age = col1.number_input("Yaş", min_value=0, max_value=150, value=30, step=1)
    selected_gender = col2.selectbox("Cinsiyet", ["Erkek", "Kadın"])
    selected_weight = col3.number_input("Kilo (kg)", min_value=20, max_value=500, value=70, step=1)
    selected_height = col4.number_input("Boy (cm)", min_value=50, max_value=300, value=170, step=1)
    selected_CH2O = col5.number_input("Günlük Su Tüketimi (ml)", min_value=0, max_value=10000, value=2000, step=100)

    # Tahmini hesapla ve göster
    if col6.button("Tahminle"):
        prediction = predict_obesity_risk(selected_age, selected_gender, selected_weight, selected_height, selected_CH2O)
        col6.metric(label="Tahmin Edilen Obezite Riski", value=prediction)
        st.balloons()





