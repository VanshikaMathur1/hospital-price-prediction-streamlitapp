import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go

# Load dataset and model
df = pd.read_csv('data/large_hospital_cost_prediction_dataset (1).csv')
model = joblib.load('models/hospital_cost_model.pkl')

# Get the columns from the trained model
train_columns = model.feature_names_in_

# Page Configurations
st.set_page_config(page_title="Live Hospital Cost Prediction", layout="wide")
st.title("🏥 Live Hospital Cost Prediction Dashboard")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Live Predictions", "Data Insights"])

# Home Page
if page == "Home":
    st.header("Welcome to the Live Hospital Cost Prediction Dashboard!")
    st.write("""This dashboard provides real-time predictions and insights for hospital costs.""")

def predict_cost(age, gender, length_of_stay, procedure_cost, medication_cost, insurance, comorbidity):
    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender_Female': [1 if gender == 'Female' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Length_of_Stay': [length_of_stay],
        'Procedure_Cost': [procedure_cost],
        'Medication_Cost': [medication_cost],
        'Insurance_Type': [insurance],
        'Discharge_Status_1': [1],  # Assuming all inputs are for discharged patients
        'Comorbidities_None': [1 if comorbidity == 'None' else 0],
        'Comorbidities_Condition1': [1 if comorbidity == 'Condition1' else 0],
        'Comorbidities_Condition2': [1 if comorbidity == 'Condition2' else 0]
    })

    # Ensure all model columns are present
    for col in train_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with 0

    # Reorder input_data to match the training data column order
    input_data = input_data[train_columns]

    # Make prediction
    return model.predict(input_data)

# Live Prediction Page
if page == "Live Predictions":
    st.header("Real-Time Predictions")
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        length_of_stay = st.number_input("Length of Stay (days)", min_value=1, value=3)
        procedure_cost = st.number_input("Procedure Cost", min_value=0.0, value=1000.0)
        medication_cost = st.number_input("Medication Cost", min_value=0.0, value=500.0)
        insurance_type = st.selectbox("Insurance Type", df['Insurance_Type'].unique())
        comorbidity = st.selectbox("Comorbidities", ["None", "Condition1", "Condition2"])  # Adjust according to your dataset
        submit = st.form_submit_button("Predict Cost")

        if submit:
            predicted_cost = predict_cost(age, gender, length_of_stay, procedure_cost, medication_cost, insurance_type, comorbidity)
            st.success(f"Predicted Hospital Cost: ${predicted_cost[0]:,.2f}")

    # Real-Time Data Feed Simulation
    if st.button("Start Real-Time Simulation"):
        st.write("Simulating live data feed...")
        for _ in range(5):
            random_age = np.random.randint(20, 90)
            random_stay = np.random.randint(1, 10)
            random_procedure_cost = np.random.uniform(500, 5000)
            random_medication_cost = np.random.uniform(100, 2000)
            random_gender = np.random.choice(["Male", "Female"])
            random_insurance = np.random.choice(df['Insurance_Type'].unique())
            random_comorbidity = np.random.choice(["None", "Condition1", "Condition2"])

            live_cost = predict_cost(random_age, random_gender, random_stay, random_procedure_cost, random_medication_cost, random_insurance, random_comorbidity)
            st.write(f"Age: {random_age}, Stay: {random_stay} days, Predicted Cost: ${live_cost[0]:,.2f}")
            time.sleep(2)

# Data Insights Page
if page == "Data Insights":
    st.header("Data Insights")
    
    # Cost Distribution
    st.subheader("Cost Distribution")
    fig = px.histogram(df, x="Total_Cost", nbins=30, title="Total Cost Distribution")
    st.plotly_chart(fig)

    # Cost by Admission Type
    st.subheader("Cost by Admission Type")
    fig = px.box(df, x="Admission_Type", y="Total_Cost", color="Admission_Type", title="Cost by Admission Type")
    st.plotly_chart(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        colorbar=dict(title='Correlation Coefficient'),
    ))
    fig.update_layout(title='Correlation Heatmap', xaxis_title='Features', yaxis_title='Features')
    st.plotly_chart(fig)
