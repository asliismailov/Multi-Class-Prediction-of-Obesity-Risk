import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Modelin yüklenmesi
model = pickle.load(open('lgbm_model_final.pkl', 'rb'))

# Veri ön işleme için hazırlık
numerik_ozellikler = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']
kategorik_ozellikler = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerik_ozellikler),
        ('cat', OneHotEncoder(), kategorik_ozellikler)
    ])

# Başlık
st.title('Obezite Tahmin Uygulaması')

# Veri giriş formu
with st.form(key='my_form'):
    st.write("Lütfen sağlık bilgilerinizi girin:")
    data = {}
    for feature in numerik_ozellikler:
        data[feature] = st.number_input(f'Enter {feature}', format="%.2f")
    for feature in kategorik_ozellikler:
        data[feature] = st.selectbox(f'Select {feature}', options=['Yes', 'No'])
    submit_button = st.form_submit_button(label='Tahmin Yap')

# Form gönderildiğinde
if submit_button:
    data_df = pd.DataFrame([data])
    data_preprocessed = preprocessor.fit_transform(data_df)
    prediction = model.predict(data_preprocessed)
    st.write(f"Tahmin edilen obezite durumu: {prediction[0]}")
