import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

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

if chart_tab.button("Grafikler"):
    col1, col2 = chart_tab.columns(2)

    with col1:
        st.header("Korelasyon Matrisi")
        st.image("korelasyon.png")

    with col2:
        st.header("Shap")
        st.image("SHAP.png")

if prediction_tab.button("Model"):

    def predict_obesity_risk(age, gender, weight, height, ch2o, fcvc, bmi):
        # Cinsiyeti sayısal değere dönüştür
        gender_dict = {"Erkek": 1, "Kadın": 0}
        gender_numeric = gender_dict[gender]

        # Medyan değerlerini kullanarak kategorik değişkenleri doldurma
        df = get_data()
        median_family_history = df['family_history_with_overweight'].median()
        median_gender = df['Gender'].median()

        # Boş veya yanlış girilen değerleri medyanlarla doldurma
        age = age if pd.notnull(age) and age >= 0 else df['Age'].median()
        weight = weight if pd.notnull(weight) and weight >= 0 else df['Weight'].median()
        height = height if pd.notnull(height) and height >= 0 else df['Height'].median()
        ch2o = ch2o if pd.notnull(ch2o) and ch2o >= 0 else df['CH2O'].median()
        fcvc = fcvc if pd.notnull(fcvc) and fcvc >= 0 else df['FCVC'].median()
        bmi = bmi if pd.notnull(bmi) and bmi >= 0 else df['BMI'].median()

        # Modelin tahmin yapabilmesi için gerekli diğer değişkenlerin hazırlanması
        data = {
            'Age': [age],
            'Gender': [gender_numeric],
            'Weight': [weight],
            'Height': [height],
            'CH2O': [ch2o],
            'FCVC': [fcvc],
            'BMI': [bmi],
            'family_history_with_overweight': [median_family_history],
            # Diğer gerekli özellikleri modele ekleyin
        }

        # Modelin tahmin yapması
        pipeline = get_pipeline()
        prediction = pipeline.predict(data)

        return prediction

    model_cont = prediction_tab.container()
    model_cont.subheader("Tahmin")
    col1, col2, col3, col4, col5, col6

