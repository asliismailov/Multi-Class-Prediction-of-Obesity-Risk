import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load

# Modelin yüklenmesi
model = load('lgbm_model_final.pkl')

# Veri ön işleme için hazırlık
numerik_ozellikler = ['Age', 'Height', 'Weight']
kategorik_ozellikler = ['Gender', 'family_history_with_overweight']

# OneHotEncoder setup with handle_unknown='ignore' to handle any unknown categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerik_ozellikler),
        ('cat', OneHotEncoder(handle_unknown='ignore'), kategorik_ozellikler)
    ], remainder='passthrough')  # Remainder için kalan sütunları da işlemek için

# Başlık
st.title('Obezite Tahmin Uygulaması')

# Veri giriş formu
with st.form(key='my_form'):
    st.write("Lütfen sağlık bilgilerinizi girin:")
    data = {}
    for feature in numerik_ozellikler:
        data[feature] = st.number_input(f'Enter {feature}', format="%.2f")
    
    # Kategorik değişkenler için medyanları kullanarak seçim yapma
    for feature in kategorik_ozellikler:
        # Medyan hesaplamak için geçerli veri setini yükle
        df = pd.read_csv('df_encoded.csv')
        median_value = df[feature].median()
        # Medyan değerine göre seçim yapma
        data[feature] = st.selectbox(f'Select {feature}', options=['Yes', 'No'], index=int(median_value))

    submit_button = st.form_submit_button(label='Tahmin Yap')

# Form gönderildiğinde
if submit_button:
    data_df = pd.DataFrame([data])
    data_preprocessed = preprocessor.fit_transform(data_df)
    prediction = model.predict(data_preprocessed)
    st.write(f"Tahmin edilen obezite durumu: {prediction[0]}")

