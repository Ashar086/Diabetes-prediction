import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the Pima Indians Diabetes dataset
@st.cache_data
def load_data():
    # URL of the Pima Indians Diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    # Column names for the dataset
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    # Load the dataset
    try:
        data = pd.read_csv(url, names=column_names)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Fallback to a sample dataset if the URL is not accessible
        return pd.DataFrame({
            'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10],
            'Glucose': [148, 85, 183, 89, 137, 116, 78, 115],
            'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0],
            'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0],
            'Insulin': [0, 0, 0, 94, 168, 0, 88, 0],
            'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31, 35.3],
            'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134],
            'Age': [50, 31, 32, 21, 33, 30, 26, 29],
            'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
        })

# Train the model
@st.cache_resource
def train_model(data):
    # Split the data into features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    return model, scaler

def predict():
    st.sidebar.header('Diabetes Prediction')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Diabetes Prediction(Only for Females Above 21 Years of Age)')
    st.markdown('This trained dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.')
    st.markdown('Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.')

    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:")
    st.markdown('Pregnancies: Number of times pregnant')

    glucose = st.number_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.number_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    dpf = st.number_input("Diabetes Pedigree Function:")
    st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


    age = st.number_input("Age:")
    st.markdown('Age: Age (years)')


    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')


    if submit:
        # Load the data and train the model
        data = load_data()
        model, scaler = train_model(data)
        
        # Scale the input features
        input_data = [[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]]
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        
        if prediction[0] == 0:
            st.write('Congratulation!',name,'You are not diabetic')
        else:
            st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
            st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')


def main():
    new_title = '<p style="font-size: 42px;">Welcome The Diabetes Prediction App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    The application is built using Streamlit  
    to demonstrate Diabetes Prediction. It performs prediction on multiple parameters
    [here](https://github.com/Ashar086/Diabetes-prediction#signal_strength-dataset).""")
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox(
        "MODE", ("About", "Predict Diabetes"))
    if choice == "Predict Diabetes":
        read_me_0.empty()
        read_me.empty()
        predict()
    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
