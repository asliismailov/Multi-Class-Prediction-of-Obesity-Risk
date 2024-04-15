import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Modeli joblib ile yükle
model = joblib.load("lgbm_model_final.pkl")

# Veri setini yükle
data = pd.read_csv("predicted_obesity_levels.csv")

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
        st.image('/Users/ugurcanodabasi/Desktop/PycharmProjects/Data_Scientist_Bootcamp/Streamlit/Asli.jpeg', width=150)
        st.markdown('**Aslı Öztürk**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/ozturk-asli/)')
        st.write('Aslı, veri bilimi ve makine öğrenmesi alanlarında uzman bir araştırmacıdır. Çeşitli projelerde büyük verilerle çalışmış ve sağlık teknolojileri konusunda deneyim kazanmıştır.')

    with col2:
        st.image('/Users/ugurcanodabasi/Desktop/PycharmProjects/Data_Scientist_Bootcamp/Streamlit/Begum.jpeg', width=150)
        st.markdown('**Begüm Baybora**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/begumbaybora/)')
        st.write('Begüm, yapay zeka ve veri analizi konularında geniş tecrübeye sahip bir yazılım mühendisidir. Eğitim teknolojileri ve sağlık sektöründe çözümler geliştirmiştir.')

    with col3:
        st.image('/Users/ugurcanodabasi/Desktop/PycharmProjects/Data_Scientist_Bootcamp/Streamlit/Ugur.jpeg', width=150)
        st.markdown('**Uğur Can Odabaşı**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/ugurcanodabasi)')
        st.write('Uğur Can, yazılım geliştirme ve sistem mühendisliği alanlarında derinlemesine bilgi sahibidir. Çeşitli endüstrilerde teknoloji çözümleri sunmuş ve teknik liderlik yapmıştır.')

# Anasayfa
def home_page():
    st.title('Anasayfa')

    # Obezite oranlarını gösteren dünya haritası (yer tutucu olarak statik bir görsel)
    st.image('/Users/ugurcanodabasi/Desktop/PycharmProjects/Data_Scientist_Bootcamp/Streamlit/Obesity_rate_(WHO,_2022).png', caption='Dünya Obezite Haritası')

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
def dynamic_graphs():
    st.title('Dinamik Grafikler')
    fig = px.histogram(data, x='Age', nbins=20, title='Yaş Dağılımı')
    st.plotly_chart(fig)


import numpy as np
import pandas as pd

# Obezite Tahmini
def predict_obesity():
    st.title('Obezite Tahmini Yap')

    # Kullanıcıdan alınacak girdiler
    gender = st.selectbox('Cinsiyetiniz', ['Erkek', 'Kadın'])
    age = st.number_input('Yaşınız', min_value=1, max_value=100, value=25)
    height = st.number_input('Boyunuz (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Kilonuz (kg)', min_value=20, max_value=200, value=70)
    family_history = st.selectbox('Ailenizde obezite var mı?', ['Evet', 'Hayır'])
    favc = st.selectbox('Yüksek kalorili yiyecek tüketim sıklığınız?', ['Hiç', 'Ara sıra', 'Sık'])
    fcvc = st.selectbox('Sebze tüketim sıklığınız?', ['Hiç', 'Ara sıra', 'Sık', 'Çok sık'])
    ncp = st.selectbox('Günlük ana öğün sayınız?', ['1', '2', '3', '4 veya daha fazla'])
    caec = st.selectbox('Aşırı yeme davranışınız?', ['Asla', 'Bazen', 'Sıklıkla', 'Her zaman'])
    smoke = st.selectbox('Sigara kullanımınız?', ['Evet', 'Hayır'])
    ch2o = st.slider('Günlük su tüketiminiz (litre)', 0.0, 5.0, 2.0)
    scc = st.selectbox('Kalori tüketimini takip ediyor musunuz?', ['Evet', 'Hayır'])
    faf = st.slider('Haftalık fiziksel aktivite sıklığınız (gün)', 0, 7, 3)
    tue = st.slider('Günlük teknoloji kullanım süreniz (saat)', 0, 24, 8)
    calc = st.selectbox('Alkol tüketim sıklığınız?', ['Asla', 'Bazen', 'Sıklıkla', 'Her zaman'])
    mtrans = st.selectbox('Genel ulaşım şekliniz?', ['Yürüyerek', 'Bisiklet', 'Toplu taşıma', 'Araba', 'Motosiklet'])

    # Kategorik değişkenleri sayısal değerlere dönüştür
    gender = 1 if gender == 'Erkek' else 0
    family_history = 1 if family_history == 'Evet' else 0
    favc = 1 if favc == 'Sık' else 0
    fcvc_mapping = {'Hiç': 0, 'Ara sıra': 1, 'Sık': 2, 'Çok sık': 3}
    fcvc = fcvc_mapping[fcvc]
    ncp_mapping = {'1': 1, '2': 2, '3': 3, '4 veya daha fazla': 4}
    ncp = ncp_mapping[ncp]
    caec_mapping = {'Asla': 0, 'Bazen': 1, 'Sıklıkla': 2, 'Her zaman': 3}
    caec = caec_mapping[caec]
    smoke = 1 if smoke == 'Evet' else 0
    scc = 1 if scc == 'Evet' else 0
    calc_mapping = {'Asla': 0, 'Bazen': 1, 'Sıklıkla': 2, 'Her zaman': 3}
    calc = calc_mapping[calc]
    mtrans_mapping = {'Yürüyerek': 0, 'Bisiklet': 1, 'Toplu taşıma': 2, 'Araba': 3, 'Motosiklet': 4}
    mtrans = mtrans_mapping[mtrans]

    # Toplam girdi vektörünü oluştur
    input_data = np.array(
        [gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans])


    # Tüm değişkenleri içeren girdi matrisini oluşturun ve eksik değerleri ekleyin (Eğer varsa)
    if input_data.shape[0] < 24:
        additional_data = np.full((24 - input_data.shape[0],), np.nan)  # Eksik özellikler için NaN doldur
        input_data = np.concatenate((input_data, additional_data))

    if st.button('Tahmin Et'):
        prediction = model.predict(input_data.reshape(1, -1))
        # Obezite durumunu açıklamak için bir sözlük kullan
        obesity_status = {
            0: 'Yetersiz Kilolu',
            1: 'Normal Kilolu',
            2: 'Fazla Kilolu Seviye I',
            3: 'Fazla Kilolu Seviye II',
            4: 'Obezite Tip I',
            5: 'Obezite Tip II',
            6: 'Obezite Tip III'
        }
        predicted_status = obesity_status.get(prediction[0], "Bilinmeyen durum")
        st.subheader(f'Tahmin edilen obezite durumu: {predicted_status}')


        # Görsel ve öneri metnini ekle
        advice_dict = {
            'Yetersiz Kilolu': ("Düzenli ve dengeli beslenmek önemlidir. Daha sık ve kaliteli besinler tüketmeye özen gösterin."),
            'Normal Kilolu': ("Mevcut sağlıklı yaşam tarzınızı sürdürün ve düzenli fiziksel aktiviteler yapın."),
            'Fazla Kilolu Seviye I': ("Düzenli egzersiz ve sağlıklı beslenme planı oluşturarak kilo vermeye çalışın."),
            'Fazla Kilolu Seviye II': ("Sağlık profesyonelleriyle görüşerek kişisel bir kilo verme planı hazırlayın."),
            'Obezite Tip I': ("Beslenme ve egzersiz alışkanlıklarınızı gözden geçirin, gerekiyorsa uzman desteği alın."),
            'Obezite Tip II': ("Ciddi sağlık riskleri için doktor kontrolü ve düzenli takip şart."),
            'Obezite Tip III': ("Hemen tıbbi yardım alın ve obeziteyle mücadele için multidisipliner bir yaklaşım benimseyin.")
        }
        advice = advice_dict[predicted_status]
        st.write(advice)


# Diğer fonksiyonlar ve Streamlit yapılandırması

# Streamlit uygulamasını yapılandır
def main():
    st.sidebar.title('Navigasyon')
    page = st.sidebar.radio('Sayfayı Seçin:', ['Hakkımızda', 'Anasayfa', 'Dinamik Grafikler', 'Obezite Tahmini'])

    if page == 'Hakkımızda':
        display_about()
    elif page == 'Anasayfa':
        home_page()
    elif page == 'Dinamik Grafikler':
        dynamic_graphs()
    elif page == 'Obezite Tahmini':
        predict_obesity()


if __name__ == "__main__":
    main()






