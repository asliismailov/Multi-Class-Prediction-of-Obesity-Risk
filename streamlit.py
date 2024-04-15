import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the model using joblib
model = joblib.load("lgbm_model_final.pkl")

# Load the dataset
data = pd.read_csv("predicted_obesity_levels.csv")

def display_about():
    st.title('About Us')
    st.write('''
    This application is developed to predict obesity status based on user health data.
    It analyzes various nutrition and lifestyle data to help assess obesity risk.
    Our project aims to provide informative content for users to better understand and manage their health.
    ''')

    st.subheader('Team Members')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('/Users/aslitopkoru/Desktop/Data\ Scientist/Proje/asli.jpeg', width=150)
        st.markdown('**Aslı Öztürk**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/ozturk-asli/)')
        st.write('Aslı is an expert researcher in data science and machine learning. She has worked with big data in various projects and gained experience in health technologies.')

    with col2:
        st.image('/Users/aslitopkoru/Desktop/Data\ Scientist/Proje/begum.jpeg', width=150)
        st.markdown('**Begüm Baybora**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/begumbaybora/)')
        st.write('Begüm is a software engineer with extensive experience in artificial intelligence and data analysis. She has developed solutions in education technologies and the health sector.')

    with col3:
        st.image('/Users/aslitopkoru/Desktop/Data\ Scientist/Proje/ugur.jpeg', width=150)
        st.markdown('**Uğur Can Odabaşı**')
        st.markdown('[LinkedIn Profile](https://www.linkedin.com/in/ugurcanodabasi)')
        st.write('Uğur Can is deeply knowledgeable in software development and system engineering. He has provided technology solutions in various industries and served as a technical leader.')

# Home Page
def home_page():
    st.title('Home Page')

    # Placeholder for a static image showing obesity rates globally
    st.image('/path/to/Obesity_rate_(WHO,_2022).png', caption='World Obesity Map')

    # Text describing obesity as a modern disease
    st.write("""
    ## Obesity: The Modern Epidemic

    Obesity is a serious health problem affecting millions of people worldwide and spreading rapidly.
    Individuals with a Body Mass Index (BMI) over 30 are classified as obese, which significantly increases
    the risk of various chronic diseases. Obesity can lead to health issues such as heart disease, type 2 diabetes,
    certain types of cancer, and musculoskeletal disorders.

    Combating obesity requires not only individual but also global efforts. Adopting healthy eating habits,
    engaging in regular physical activity, and making healthy lifestyle choices play a key role in fighting obesity.
    Societies and governments should develop and implement policies to facilitate access to healthy foods,
    promote physical activity, and increase health education.
    """)

# Dynamic Graphs
def dynamic_graphs():
    st.title('Dynamic Graphs')
    fig = px.histogram(data, x='Age', nbins=20, title='Age Distribution')
    st.plotly_chart(fig)


# Obesity Prediction
def predict_obesity():
    st.title('Predict Obesity')

    # User inputs
    gender = st.selectbox('Your Gender', ['Male', 'Female'])
    age = st.number_input('Your Age', min_value=1, max_value=100, value=25)
    height = st.number_input('Your Height (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Your Weight (kg)', min_value=20, max_value=200, value=70)
    family_history = st.selectbox('Family History of Obesity?', ['Yes', 'No'])
    favc = st.selectbox('Frequency of High Caloric Food Consumption?', ['Never', 'Sometimes', 'Often'])
    fcvc = st.selectbox('Frequency of Vegetable Consumption?', ['Never', 'Sometimes', 'Often', 'Very Often'])
    ncp = st.selectbox('Number of Main Meals per day?', ['1', '2', '3', '4 or more'])
    caec = st.selectbox('Excessive Eating Behavior?', ['Never', 'Sometimes', 'Often', 'Always'])
    smoke = st.selectbox('Do You Smoke?', ['Yes', 'No'])
    ch2o = st.slider('Daily Water Intake (liters)', 0.0, 5.0, 2.0)
    scc = st.selectbox('Do You Monitor Caloric Intake?', ['Yes', 'No'])
    faf = st.slider('Weekly Frequency of Physical Activity (days)', 0, 7, 3)
    tue = st.slider('Daily Technology Use Time (hours)', 0, 24, 8)
    calc = st.selectbox('Alcohol Consumption Frequency?', ['Never', 'Sometimes', 'Often', 'Always'])
    mtrans = st.selectbox('Your Main Transportation Method?', ['Walking', 'Cycling', 'Public Transport', 'Car', 'Motorbike'])

    # Convert categorical variables to numerical values
    gender = 1 if gender == 'Male' else 0
    family_history = 1 if family_history == 'Yes' else 0
    favc = 1 if favc == 'Often' else 0
    fcvc_mapping = {'Never': 0, 'Sometimes': 1, 'Often': 2, 'Very Often': 3}
    fcvc = fcvc_mapping[fcvc]
    ncp_mapping = {'1': 1, '2': 2, '3': 3, '4 or more': 4}
    ncp = ncp_mapping[ncp]
    caec_mapping = {'Never': 0, 'Sometimes': 1, 'Often': 2, 'Always': 3}
    caec = caec_mapping[caec]
    smoke = 1 if smoke == 'Yes' else 0
    scc = 1 if scc == 'Yes' else 0
    calc_mapping = {'Never': 0, 'Sometimes': 1, 'Often': 2, 'Always': 3}
    calc = calc_mapping[calc]
    mtrans_mapping = {'Walking': 0, 'Cycling': 1, 'Public Transport': 2, 'Car': 3, 'Motorbike': 4}
    mtrans = mtrans_mapping[mtrans]

    # Create the input vector
    input_data = np.array(
        [gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans])

    # Create the input matrix with all features and add missing values if any
    if input_data.shape[0] < 24:
        additional_data = np.full((24 - input_data.shape[0],), np.nan)  # Fill with NaN for missing features
        input_data = np.concatenate((input_data, additional_data))

    if st.button('Predict'):
        prediction = model.predict(input_data.reshape(1, -1))
        # Use a dictionary to describe the obesity status
        obesity_status = {
            0: 'Underweight',
            1: 'Normal Weight',
            2: 'Overweight Level I',
            3: 'Overweight Level II',
            4: 'Obesity Type I',
            5: 'Obesity Type II',
            6: 'Obesity Type III'
        }
        predicted_status = obesity_status.get(prediction[0], "Unknown status")
        st.subheader(f'Predicted obesity status: {predicted_status}')

        # Add visual and advice text
        advice_dict = {
            'Underweight': ("It's important to eat regularly and balanced. Try to consume more frequent and quality meals."),
            'Normal Weight': ("Maintain your current healthy lifestyle and engage in regular physical activities."),
            'Overweight Level I': ("Try to lose weight by creating a regular exercise and healthy eating plan."),
            'Overweight Level II': ("Consult with healthcare professionals to create a personalized weight loss plan."),
            'Obesity Type I': ("Review your eating and exercise habits, seek expert support if necessary."),
            'Obesity Type II': ("Serious health risks, require doctor monitoring and regular check-ups."),
            'Obesity Type III': ("Seek immediate medical attention and adopt a multidisciplinary approach to combat obesity.")
        }
        advice = advice_dict[predicted_status]
        st.write(advice)


# Other functions and Streamlit configuration

# Configure the Streamlit application
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select a Page:', ['About Us', 'Home', 'Dynamic Graphs', 'Obesity Prediction'])

    if page == 'About Us':
        display_about()
    elif page == 'Home':
        home_page()
    elif page == 'Dynamic Graphs':
        dynamic_graphs()
    elif page == 'Obesity Prediction':
        predict_obesity()


if __name__ == "__main__":
    main()






