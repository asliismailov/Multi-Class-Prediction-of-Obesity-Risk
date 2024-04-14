import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Özel bir önbellek yöneticisi tanımlama
custom_cache = st.cache(allow_output_mutation=True, persist=True, suppress_st_warning=True, show_spinner=False)

st.set_page_config(layout="wide", page_title="Obezite Riskinin Çok Sınıflı Tahmini", page_icon="🎷")

@st.cache
def get_data():
    dataframe = pd.read_csv('predicted_obesity_levels.csv')
    return dataframe

# Modeli yükle
@st.cache
def get_pipeline():
    pipeline = joblib.load('lgbm_model_final.pkl')
    return pipeline

main_tab, chart_tab, prediction_tab = st.columns(3)

if main_tab.button("Ana Sayfa"):
    main_tab.write("""Bu projenin amacı, bireylerde kardiyovasküler hastalıklarla ilişkili obezite riskini tahmin etmek için çeşitli faktörleri kullanmaktır. Kardiyovasküler hastalıklar, dünya genelinde sağlık sorunlarının önde gelen nedenlerinden biri olarak kabul edilmektedir. Bu hastalıkların birçoğu obezite ile doğrudan ilişkilidir. Bu nedenle, obeziteyi öngörmek ve bu konuda farkındalık yaratmak önemlidir.""")

    main_tab.write("""Veri Seti ve Hedef
    Bu projede kullanılan veri seti, bireylerin demografik bilgilerini, yaşam tarzı alışkanlıklarını ve fizyolojik ölçümlerini içerir. Ölçümler arasında boy, kilo, günlük su tüketimi, fiziksel aktivite düzeyi gibi faktörler bulunmaktadır. Veri setindeki her bir satır, bir bireyi temsil eder ve bu bireylerin obezite durumları "NObeyesdad" sütununda belirtilmiştir.""")

    # TAVSİYE: Veri setinin bir kısmı eklenebilir

    main_tab.write("""Kullanılan Algoritmalar
    Bu proje, LightGBM makine öğrenimi modeli kullanmaktadır. LightGBM, yüksek performanslı ve hızlı bir gradyan arttırma (gradient boosting) algoritmasıdır. Bu algoritma, veri setindeki örüntüleri öğrenerek ve karmaşık ilişkileri modelleyerek obezite riskini tahmin etmek için kullanılır.""")

    main_tab.write("""Uygulama: Streamlit ile Model Tahmini
    Bu projede, geliştirilen modelin kullanıcı dostu bir arayüz ile sunulması amaçlanmıştır. Streamlit adlı Python kütüphanesi, basit ve etkileşimli web uygulamaları oluşturmayı sağlar. Bu projede, geliştirilen LightGBM modeli Streamlit arayüzü ile entegre edilmiştir.
    Kullanıcılar, arayüz üzerinden bireysel özellikleri girebilir ve modele besleyerek obezite risk tahminini alabilirler. Bu tahminler, bireylerin normal kilolu, aşırı kilolu, obez veya aşırı obez olma riskini belirtir.""")

if chart_tab.button("Grafikler"):
    chart_tab.write("Grafikler sekmesine hoş geldiniz!")
    # Buraya grafiklerinizi ekleme işlemleri gelecek

if prediction_tab.button("Model"):
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
            feature_values = [selected_age, selected_height, selected_weight]  # Bu liste modelin beklediği özelliklerle doldurulmalıdır
            feature_names = ['age', 'height', 'weight']  # Bu da özellik isimleriyle doldurulmalıdır
            input_data = pd.DataFrame([feature_values], columns=feature_names)
            
            # Tahmin yap
            pipeline = get_pipeline()  # Modelinizi yükleyin
            prediction = pipeline.predict(input_data)[0]
            
            # Tahmini göster
            st.write(f'Tahmin edilen obezite seviyesi: {prediction}')
