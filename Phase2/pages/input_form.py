import streamlit as st
import pandas as pd
import pickle
import sys
import xgboost
from joblib import load


# Set page config MUST BE FIRST
st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")

# Custom CSS (keep your existing styling here)
st.markdown("""
    <style>
    /* Your existing CSS styles */
    </style>
    """, unsafe_allow_html=True)

# Load the model with proper error handling
try:
    model = load("model/depi_xgb.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()




# Verify model is loaded
if model is None:
    st.error("Failed to load the prediction model. Please check the model file.")
    st.stop()


# Page header
st.title("Diabetes Risk Assessment Tool")
st.markdown("""
Please fill in your health information below to assess your diabetes risk.
All fields are required.
""")

# Create form
with st.form("diabetes_risk_form"):
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.selectbox(
            "Age Group",
            options=[(1, "18-24"), (2, "25-29"), (3, "30-34"), 
                    (4, "35-39"), (5, "40-44"), (6, "45-49"), 
                    (7, "50-54"), (8, "55-59"), (9, "60-64"), 
                    (10, "65-69"), (11, "70-74"), (12, "75-79"), 
                    (13, "80+")],
            format_func=lambda x: x[1],
            index=6
        )[0]
        
        sex = st.radio("Sex", options=[(0, "Female"), (1, "Male")], format_func=lambda x: x[1])[0]
        
        education = st.selectbox(
            "Education Level",
            options=[(1, "Never attended school"), 
                    (2, "Elementary school"), 
                    (3, "Some high school"),
                    (4, "High school graduate"),
                    (5, "Some college"),
                    (6, "College graduate")],
            format_func=lambda x: x[1],
            index=3
        )[0]
        
        income = st.selectbox(
            "Income Level",
            options=[(1, "<$10,000"), 
                    (2, "$10,000-$15,000"),
                    (3, "$15,000-$20,000"),
                    (4, "$20,000-$25,000"),
                    (5, "$25,000-$35,000"),
                    (6, "$35,000-$50,000"),
                    (7, "$50,000-$75,000"),
                    (8, ">$75,000")],
            format_func=lambda x: x[1],
            index=4
        )[0]
    
    with col2:
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1
        )
        
        # General Health - Corrected version
        gen_hlth_options = [
            (1, "Excellent"), 
            (2, "Very good"), 
            (3, "Good"),
            (4, "Fair"),
            (5, "Poor")
        ]
        gen_hlth = st.select_slider(
            "General Health",
            options=[x[0] for x in gen_hlth_options],
            value=3,
            format_func=lambda x: dict(gen_hlth_options)[x]
        )
        
        phys_hlth = st.slider(
            "Number of days with physical health issues in past 30 days",
            min_value=0,
            max_value=30,
            value=0
        )
        
        ment_hlth = st.slider(
            "Number of days with mental health issues in past 30 days",
            min_value=0,
            max_value=30,
            value=0
        )
    
    st.header("Health Conditions")
    col3, col4 = st.columns(2)
    
    with col3:
        high_bp = st.checkbox("High blood pressure", value=False)
        high_chol = st.checkbox("High cholesterol", value=False)
        chol_check = st.checkbox("Had cholesterol check in past 5 years", value=True)
        smoker = st.checkbox("Smoked at least 100 cigarettes in life", value=False)
        stroke = st.checkbox("Ever had a stroke", value=False)
    
    with col4:
        heart_disease = st.checkbox("Heart disease or heart attack", value=False)
        diff_walk = st.checkbox("Difficulty walking or climbing stairs", value=False)
        phys_activity = st.checkbox("Physical activity in past 30 days", value=True)
        fruits = st.checkbox("Consume fruit daily", value=False)
        veggies = st.checkbox("Consume vegetables daily", value=False)
    
    st.header("Healthcare Access")
    hvy_alcohol = st.checkbox("Heavy alcohol consumption (adult men >14 drinks/week, women >7)", value=False)
    any_healthcare = st.checkbox("Have any healthcare coverage", value=True)
    no_doc_cost = st.checkbox("Couldn't see doctor in past year due to cost", value=False)
    
    # Submit button
    submitted = st.form_submit_button("Assess Diabetes Risk")
    
    if submitted:
        # Prepare input data (same as before)
        input_data = {
            'HighBP': int(high_bp),
            'HighChol': int(high_chol),
            'CholCheck': int(chol_check),
            'BMI': bmi,
            'Smoker': int(smoker),
            'Stroke': int(stroke),
            'HeartDiseaseorAttack': int(heart_disease),
            'PhysActivity': int(phys_activity),
            'Fruits': int(fruits),
            'Veggies': int(veggies),
            'HvyAlcoholConsump': int(hvy_alcohol),
            'AnyHealthcare': int(any_healthcare),
            'NoDocbcCost': int(no_doc_cost),
            'GenHlth': gen_hlth,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth,
            'DiffWalk': int(diff_walk),
            'Sex': sex,
            'Age': age,
            'Education': education,
            'Income': income
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction with your actual model
        try:
            y_hat = model.predict(input_df)
            prediction_prob = model.predict_proba(input_df)[0][1]
            
            # Store results
       
            st.session_state['results'] = {
                'input_data': input_data,
                'prediction': y_hat[0],
                'prediction_prob': prediction_prob,
                'prediction_made' : True,
            }
            
            st.success("Assessment completed successfully!")
            st.switch_page("pages/prediction.py") 
            

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


