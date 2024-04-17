import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modelin yüklenmesi
model = joblib.load('Miuul_Final.pkl')

def display_about():
    st.title('Hakkımızda')
    st.write('''
    Bu uygulama, kullanıcıların sağlık verileri üzerinden obezite durumunu tahmin etmek amacıyla geliştirilmiştir.
    Uygulama, çeşitli beslenme ve yaşam tarzı verilerini analiz ederek obezite riskini değerlendirmeye yardımcı olur.
    Projemiz, kullanıcıların sağlıklarını daha iyi anlamaları ve yönetmeleri için bilgilendirici içerikler sunmayı hedeflemektedir.
    ''')

    st.subheader('Takım Üyeleri')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('asli.jpeg', width=150)
        st.markdown('**Aslı Öztürk**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/ozturk-asli/)')
        st.write('Aslı, veri bilimi ve makine öğrenmesi alanlarında uzman bir araştırmacıdır. Çeşitli projelerde büyük verilerle çalışmış ve sağlık teknolojileri konusunda deneyim kazanmıştır.')

    with col2:
        st.image('begum.jpeg', width=150)
        st.markdown('**Begüm Baybora**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/begumbaybora/)')
        st.write('Begüm, yapay zeka ve veri analizi konularında geniş tecrübeye sahip bir yazılım mühendisidir. Eğitim teknolojileri ve sağlık sektöründe çözümler geliştirmiştir.')

    with col3:
        st.image('ugur.jpeg', width=150)
        st.markdown('**Uğur Can Odabaşı**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/ugurcanodabasi)')
        st.write('Uğur Can, yazılım geliştirme ve sistem mühendisliği alanlarında derinlemesine bilgi sahibidir. Çeşitli endüstrilerde teknoloji çözümleri sunmuş ve teknik liderlik yapmıştır.')

# Anasayfa
def home_page():
    st.title('Anasayfa')

    # Obezite oranlarını gösteren dünya haritası (yer tutucu olarak statik bir görsel)
    st.image('Obesity_rate_(WHO,_2022).png', caption='Dünya Obezite Haritası')

    # Obezitenin çağın hastalığı olduğunu anlatan metin
    st.write("""
    ## Obezite: Çağımızın Hastalığı

    Obezite, dünya genelinde milyonlarca insanı etkileyen ve hızla yayılan ciddi bir sağlık sorunudur. 
    Vücut kitle indeksi (VKİ) 30'un üzerinde olan bireyler obez olarak sınıflandırılır ve bu durum, 
    çeşitli kronik hastalıkların riskini önemli ölçüde artırır. Obezite; kalp hastalıkları, tip 2 diyabet, 
    bazı kanser türleri ve kas-iskelet sistemi bozuklukları gibi sağlık sorunlarına yol açabilir.

    Obeziteyle mücadele, sadece bireysel değil, aynı zamanda küresel bir çaba gerektirir. 
    Sağlıklı beslenme alışkanlıkları edinmek, düzenli fiziksel aktivite yapmak ve sağlıklı yaşam tarzı seçimleri,
    obeziteyle mücadelede kilit rol oynar. Toplumlar ve hükümetler, sağlıklı gıdalara erişimi kolaylaştırmak, 
    fiziksel aktiviteyi teşvik etmek ve sağlık eğitimini artırmak için politikalar geliştirmeli ve uygulamalıdır.
    """)


# Dinamik Grafikler

# Obezite Tahmini

# Kullanıcı girişlerini al
st.title('Obezite Durumu Tahmini')

# Modelin en çok etkilendiği özellikler
weight = st.number_input('Kilo (kg):', min_value=0.0, format="%.2f")
age = st.number_input('Yaş:', min_value=0, max_value=120)
height = st.number_input('Boy (metre):', min_value=0.0, format="%.2f")  # Metreye çevir
faf = st.slider('Fiziksel aktivite frekansı (haftada kaç gün):', 0, 7, 1)
ch2o = st.slider('Günlük su tüketimi (Litre):', 0.0, 5.0, 1.0)
tue = st.slider('Teknoloji kullanım süresi (günde kaç saat):', 0, 24, 3)
fcvc = st.slider('Günde kaç kez sebze tüketiyorsunuz?', 0, 5, 2)
ncp = st.slider('Günlük ana öğün sayısı:', 1, 5, 3)
gender = st.selectbox('Cinsiyet:', ['Kadın', 'Erkek'])
family_history_with_overweight = st.selectbox('Ailede aşırı kilolu birey var mı?', ['Evet', 'Hayır'])

# BMI hesaplama
bmi = weight / (height ** 2)

# Varsayılan değerlerin belirlenmesi ve kullanıcı girişlerinin entegrasyonu
default_values = {
    "CALC_Always": 0,
    "MTRANS_Walking": 0,
    "MTRANS_Public_Transportation": 1,
    "MTRANS_Motorbike": 0,
    "MTRANS_Bike": 0,
    "MTRANS_Automobile": 0,
    "CALC_no": 1,
    "CALC_Sometimes": 0,
    "CALC_Frequently": 0,
    "CAEC_no": 0,
    "CAEC_Sometimes": 1,
    "CAEC_Frequently": 0,
    "CAEC_Always": 0,
    "FAVC": 1,
    "SMOKE": 0,
    "SCC": 0,
    "BMI": bmi  # BMI is calculated based on user input
}

# Kullanıcı girişlerini ve varsayılan değerleri birleştirme
user_input = {
    'Weight': weight,
    'Age': age,
    'Height': height,  # height is already in meters
    'FAF': faf,
    'CH2O': ch2o,
    'TUE': tue,
    'FCVC': fcvc,
    'NCP': ncp,
    'Gender': 1 if gender == 'Erkek' else 0,  # Binary conversion for gender
    'family_history_with_overweight': 1 if family_history_with_overweight == 'Evet' else 0,
}

full_data = {**default_values, **user_input}

# Veri çerçevesi oluşturuluyor
input_df = pd.DataFrame([full_data])

# Tahmin yapılıyor
if st.button('Tahmin Yap'):
    prediction = model.predict(input_df)
    st.write(f'Tahmin edilen Obezite Durumu: {prediction[0]}')
