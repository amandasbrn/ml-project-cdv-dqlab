import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

# set halaman judul
st.set_page_config(page_title="Machine Learning Dashboard", layout="wide")
st.write("""
# :bar_chart: :heart: Predicting Cardiovascular Disease

Welcome to my machine learning dashboard to predict if someone has cardiovascular disease or not.
This dashboard is created by : [Dira Amanda](https://www.linkedin.com/in/dira-amanda/)
""")
# st.markdown("""<hr style="height:30px;border:none;color:#F0EEEE;background-color:#F0EEEE;" /> """, unsafe_allow_html=True)
st.sidebar.header('Fill the data below.')

def heart():
    st.write("""
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            st.sidebar.subheader('Chest Pain Type')
            cp = st.sidebar.slider('input chest pain type', 1,4,2)
            if cp == 1:
                wcp = "angina chest pain"
            elif cp == 2:
                wcp = "unstable chest pain"
            elif cp == 3:
                wcp = "severe unstable chest pain"
            else:
                wcp = "not related to heart problems"
            st.sidebar.write("The type of chest pain felt by the patient is", wcp)
            st.sidebar.write("---")
            st.sidebar.subheader('Maximum heart rate achieved')
            thalach = st.sidebar.slider("input heart rate", 71, 202, 80)
            st.sidebar.write("---")
            st.sidebar.subheader('Slope of the peak exercise ST segment from Electrocardiogram (EKG)')
            slope = st.sidebar.slider("input slope", 0, 2, 1)
            st.sidebar.write("---")
            st.sidebar.subheader('ST depression induced by exercise relative to rest')
            oldpeak = st.sidebar.slider("input oldpeak", 0.0, 6.2, 1.0)
            st.sidebar.write("---")
            st.sidebar.subheader('Exercise induced angina')
            exang = st.sidebar.slider("Input yes(1) or no (0)", 0, 1, 1)
            st.sidebar.write("---")
            st.sidebar.subheader('Number of major vessels')
            ca = st.sidebar.slider("number of major vessels (0-3) colored by flourosopy", 0, 3, 1)
            st.sidebar.write("---")
            st.sidebar.subheader('Thalassemia test result')
            thal = st.sidebar.slider("1: normal. 2: permanent defect in thalassemia. 3: a defect that can be broadcast in thalassemia.", 1, 3, 1)
            st.sidebar.write("---")
            st.sidebar.subheader('Gender')
            sex = st.sidebar.selectbox("Gender of the patient", ('Female', 'Male'))
            if sex == "Female":
                sex = 0
            else:
                sex = 1
            st.sidebar.subheader('Patient Age')
            age = st.sidebar.slider("Input age", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features

    input_df = user_input_features()
    img = Image.open("/Users/diraamandas/Desktop/DQLAB_capstone/heart_disease.jpeg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        df = input_df
        st.write("Inputted data:")
        st.write(df)
        loaded_model = pickle.load(open('/Users/diraamandas/Desktop/DQLAB_capstone/model_with_mlp.pkl','rb'))
        prediction = loaded_model.predict(df)
        #if prediction == 1.0:
        #    result = ['has cardiovascular disease. Seek professional help immediately.']
        #elif prediction == 0.0:
        #    result = ['has no cardiovascular disease']
        st.subheader('Prediction: ')
        #output = str(result[0])
        with st.spinner('Please wait'):
            time.sleep(4)
            if prediction == 1.0:
                st.success("This person has no cardiovascular disease")
            elif prediction == 0.0:
                st.error("This person has cardiovascular disease. Seek professional help immediately.")

heart()