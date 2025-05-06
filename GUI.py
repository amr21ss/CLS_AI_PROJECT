# import pickle
# import streamlit as st
# import numpy as np

# # Load the diabetes model
# try:
#     with open(r'D:\Gam3a\CLS\project\Data\diabetes_predictions.sav', 'rb') as file:
#         model = pickle.load(file)
    
#     # Verify the model has predict method
#     if not hasattr(model, "predict"):
#         st.error("🔴 The loaded model does not have a 'predict' function.")
#         st.stop()
# except FileNotFoundError:
#     st.error("🔴 Model file not found! Please check the path.")
#     st.stop()
# except Exception as e:
#     st.error(f"🔴 Error loading model: {e}")
#     st.stop()

# # App configuration
# st.set_page_config(page_title="🩺 Diabetes Risk Assessment", layout="centered")

# # Custom CSS styling
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: gray;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
#     .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
#         color: black;
#     }
#     .stSelectbox label, .stSlider label, .stRadio label {
#         color: black !important;
#     }
#     .highlight {
#         background-color: #ffd700;
#         padding: 0.2em;
#         border-radius: 3px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title and header
# st.title("🩺 Diabetes Risk Assessment Tool")
# st.markdown("---")
# st.markdown("### 📋 Complete the Health Assessment Below")

# # Input sections
# st.markdown("#### 🩸 Blood Pressure & Cholesterol")
# col1, col2, col3 = st.columns(3)
# with col1:
#     high_bp = st.radio("High Blood Pressure", ["No", "Yes"], key="highbp")
# with col2:
#     high_chol = st.radio("High Cholesterol", ["No", "Yes"], key="highchol")
# with col3:
#     chol_check = st.radio("Cholesterol Check in Past 5 Years", ["No", "Yes"], key="cholcheck")

# st.markdown("#### 🏋️ Lifestyle Factors")
# col1, col2, col3 = st.columns(3)
# with col1:
#     smoker = st.radio("Smoker", ["No", "Yes"], key="smoker")
# with col2:
#     phys_activity = st.radio("Physical Activity", ["No", "Yes"], key="physactivity")
# with col3:
#     hvy_alcohol = st.radio("Heavy Alcohol Consumption", ["No", "Yes"], key="alcohol")

# st.markdown("#### 🥗 Diet Habits")
# col1, col2 = st.columns(2)
# with col1:
#     fruits = st.radio("Fruit Consumption (≥1/day)", ["No", "Yes"], key="fruits")
# with col2:
#     veggies = st.radio("Vegetable Consumption (≥1/day)", ["No", "Yes"], key="veggies")

# st.markdown("#### 🏥 Health History")
# col1, col2, col3 = st.columns(3)
# with col1:
#     stroke = st.radio("Ever Had Stroke", ["No", "Yes"], key="stroke")
# with col2:
#     heart_disease = st.radio("Heart Disease/Attack", ["No", "Yes"], key="heartdisease")
# with col3:
#     diff_walk = st.radio("Difficulty Walking", ["No", "Yes"], key="diffwalk")

# st.markdown("#### 🩺 General Health Metrics")
# col1, col2 = st.columns(2)
# with col1:
#     bmi = st.slider("Body Mass Index (BMI)", 20, 60, 25, key="bmi")
# with col2:
#     gen_hlth = st.selectbox(
#         "General Health Rating", 
#         ["Excellent", "Very Good", "Good", "Fair", "Poor"], 
#         key="genhlth"
#     )

# st.markdown("#### 😔 Mental & Physical Health")
# col1, col2 = st.columns(2)
# with col1:
#     ment_hlth = st.slider(
#         "Mental Health Days (past 30 days)", 
#         0, 30, 0, 
#         help="Number of days with poor mental health"
#     )
# with col2:
#     phys_hlth = st.slider(
#         "Physical Health Days (past 30 days)", 
#         0, 30, 0,
#         help="Number of days with poor physical health"
#     )

# st.markdown("#### 🏥 Healthcare Access")
# col1, col2 = st.columns(2)
# with col1:
#     any_healthcare = st.radio("Has Healthcare Coverage", ["No", "Yes"], key="healthcare")
# with col2:
#     no_doc_cost = st.radio("Couldn't See Doctor Due to Cost", ["No", "Yes"], key="nodoccost")

# st.markdown("#### 👤 Demographic Information")
# col1, col2, col3 = st.columns(3)
# with col1:
#     sex = st.radio("Sex", ["Male", "Female"], key="sex")
# with col2:
#     age = st.slider("Age", 0, 100, 30, key="age")
# with col3:
#     education = st.selectbox(
#         "Education Level", 
#         [
#             "Never attended school",
#             "Elementary school",
#             "Some high school",
#             "High school graduate",
#             "Some college",
#             "College graduate"
#         ],
#         key="education"
#     )

# income = st.selectbox(
#     "Annual Household Income", 
#     [
#         "Less than $10,000",
#         "$10,000-$15,000",
#         "$15,000-$20,000",
#         "$20,000-$25,000",
#         "$25,000-$35,000",
#         "$35,000-$50,000",
#         "$50,000-$75,000",
#         "$75,000 or more"
#     ],
#     key="income"
# )

# # Convert all inputs to model format
# def prepare_inputs():
#     # Map categorical variables to numerical (keep these the same)
#     gen_hlth_map = {"Excellent":1, "Very Good":2, "Good":3, "Fair":4, "Poor":5}
#     education_map = {
#         "Never attended school": 1,
#         "Elementary school": 2,
#         "Some high school": 3,
#         "High school graduate": 4,
#         "Some college": 5,
#         "College graduate": 6
#     }
#     income_map = {
#         "Less than $10,000": 1,
#         "$10,000-$15,000": 2,
#         "$15,000-$20,000": 3,
#         "$20,000-$25,000": 4,
#         "$25,000-$35,000": 5,
#         "$35,000-$50,000": 6,
#         "$50,000-$75,000": 7,
#         "$75,000 or more": 8
#     }
    
#     # REORDERED to match model's training data
#     features = [
#         income_map[income],                  # Income
#         education_map[education],            # Education
#         age,                                 # Age
#         1 if sex == "Female" else 0,         # Sex
#         1 if diff_walk == "Yes" else 0,      # DiffWalk
#         phys_hlth,                          # PhysHlth
#         ment_hlth,                          # MentHlth
#         gen_hlth_map[gen_hlth],             # GenHlth
#         1 if no_doc_cost == "Yes" else 0,    # NoDocbcCost
#         1 if any_healthcare == "Yes" else 0, # AnyHealthcare
#         1 if hvy_alcohol == "Yes" else 0,    # HvyAlcoholConsump
#         1 if veggies == "Yes" else 0,        # Veggies
#         1 if fruits == "Yes" else 0,         # Fruits
#         1 if phys_activity == "Yes" else 0,  # PhysActivity
#         1 if heart_disease == "Yes" else 0,  # HeartDiseaseorAttack
#         1 if stroke == "Yes" else 0,         # Stroke
#         1 if smoker == "Yes" else 0,         # Smoker
#         bmi,                                # BMI
#         1 if chol_check == "Yes" else 0,     # CholCheck
#         1 if high_chol == "Yes" else 0,      # HighChol
#         1 if high_bp == "Yes" else 0         # HighBP
#     ]
    
#     return np.array(features).reshape(1, -1)

# # Prediction button
# st.markdown("---")
# if st.button('Assess Diabetes Risk', key='predict_button'):
#     try:
#         input_data = prepare_inputs()
#         prediction = model.predict(input_data)
#         prediction_proba = model.predict_proba(input_data)
        
#         # Display results
#         if prediction[0] == 0:
#             st.success("🎉 **Low Risk**: You're unlikely to have diabetes.")
#         elif prediction[0] == 1:
#             st.warning("⚠️ **Borderline Risk**: You may be at risk for diabetes.")
#         else:
#             st.error("🛑 **High Risk**: You may have diabetes.")
        
#         # Show probabilities
#         st.markdown(f"**Probability Breakdown:**")
#         st.markdown(f"- No Diabetes: {prediction_proba[0][0]*100:.1f}%")
#         st.markdown(f"- Prediabetes/Borderline: {prediction_proba[0][1]*100:.1f}%")
#         st.markdown(f"- Diabetes: {prediction_proba[0][2]*100:.1f}%")
        
#         # Recommendations based on risk level
#         st.markdown("---")
#         st.markdown("### 📝 Recommendations")
#         if prediction[0] == 0:
#             st.markdown("""
#             - Maintain your healthy lifestyle
#             - Continue regular check-ups
#             - Monitor blood sugar levels annually
#             """)
#         elif prediction[0] == 1:
#             st.markdown("""
#             - Consider more frequent blood sugar monitoring
#             - Increase physical activity
#             - Improve dietary habits
#             - Consult with a healthcare provider
#             """)
#         else:
#             st.markdown("""
#             - **Consult a doctor immediately**
#             - Begin regular blood sugar monitoring
#             - Consider dietary changes
#             - Increase physical activity
#             - Follow medical advice for diabetes management
#             """)
            
#     except Exception as e:
#         st.error(f"🔴 Error during prediction: {e}")

# # Footer
# st.markdown("---")
# st.markdown("### ℹ️ About This Tool")
# st.markdown("""
# This assessment predicts diabetes risk based on CDC behavioral risk factors. 
# It's not a substitute for professional medical advice.
# """)
# st.markdown("---")
# st.markdown("📊 **Model Information**: Random Forest Classifier trained on diabetes risk factors")



###############################################################################################

# import pickle
# import streamlit as st
# import numpy as np

# # MUST be the first Streamlit command
# st.set_page_config(page_title="🩺 Diabetes Risk Assessment", layout="centered")

# # Load the diabetes model
# try:
#     with open(r'D:\Gam3a\CLS\project\Data\diabetes_predictions.sav', 'rb') as file:
#         model = pickle.load(file)
    
#     # Verify the model has predict method
#     if not hasattr(model, "predict"):
#         st.error("🔴 The loaded model does not have a 'predict' function.")
#         st.stop()
        
#     # Debug information
#     if hasattr(model, 'feature_names_in_'):
#         st.sidebar.write("Model expects features in this order:")
#         st.sidebar.write(model.feature_names_in_)
#     elif hasattr(model, 'n_features_in_'):
#         st.sidebar.write(f"Model expects {model.n_features_in_} features")

# except FileNotFoundError:
#     st.error("🔴 Model file not found! Please check the path.")
#     st.stop()
# except Exception as e:
#     st.error(f"🔴 Error loading model: {e}")
#     st.stop()

# # Custom CSS styling
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #f0f2f6;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
#     .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
#         color: black;
#     }
#     .stSelectbox label, .stSlider label, .stRadio label {
#         color: black !important;
#     }
#     .highlight {
#         background-color: #ffd700;
#         padding: 0.2em;
#         border-radius: 3px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title and header
# st.title("🩺 Diabetes Risk Assessment Tool")
# st.markdown("---")
# st.markdown("### 📋 Complete the Health Assessment Below")

# # Input sections
# st.markdown("#### 🩸 Blood Pressure & Cholesterol")
# col1, col2, col3 = st.columns(3)
# with col1:
#     high_bp = st.radio("High Blood Pressure", ["No", "Yes"], key="highbp")
# with col2:
#     high_chol = st.radio("High Cholesterol", ["No", "Yes"], key="highchol")
# with col3:
#     chol_check = st.radio("Cholesterol Check in Past 5 Years", ["No", "Yes"], key="cholcheck")

# st.markdown("#### 🏋️ Lifestyle Factors")
# col1, col2, col3 = st.columns(3)
# with col1:
#     smoker = st.radio("Smoker", ["No", "Yes"], key="smoker")
# with col2:
#     phys_activity = st.radio("Physical Activity", ["No", "Yes"], key="physactivity")
# with col3:
#     hvy_alcohol = st.radio("Heavy Alcohol Consumption", ["No", "Yes"], key="alcohol")

# st.markdown("#### 🥗 Diet Habits")
# col1, col2 = st.columns(2)
# with col1:
#     fruits = st.radio("Fruit Consumption (≥1/day)", ["No", "Yes"], key="fruits")
# with col2:
#     veggies = st.radio("Vegetable Consumption (≥1/day)", ["No", "Yes"], key="veggies")

# st.markdown("#### 🏥 Health History")
# col1, col2, col3 = st.columns(3)
# with col1:
#     stroke = st.radio("Ever Had Stroke", ["No", "Yes"], key="stroke")
# with col2:
#     heart_disease = st.radio("Heart Disease/Attack", ["No", "Yes"], key="heartdisease")
# with col3:
#     diff_walk = st.radio("Difficulty Walking", ["No", "Yes"], key="diffwalk")

# st.markdown("#### 🩺 General Health Metrics")
# col1, col2 = st.columns(2)
# with col1:
#     bmi = st.slider("Body Mass Index (BMI)", 20, 60, 25, key="bmi")
# with col2:
#     gen_hlth = st.selectbox(
#         "General Health Rating", 
#         ["Excellent", "Very Good", "Good", "Fair", "Poor"], 
#         key="genhlth"
#     )

# st.markdown("#### 😔 Mental & Physical Health")
# col1, col2 = st.columns(2)
# with col1:
#     ment_hlth = st.slider(
#         "Mental Health Days (past 30 days)", 
#         0, 30, 0, 
#         help="Number of days with poor mental health"
#     )
# with col2:
#     phys_hlth = st.slider(
#         "Physical Health Days (past 30 days)", 
#         0, 30, 0,
#         help="Number of days with poor physical health"
#     )

# st.markdown("#### 🏥 Healthcare Access")
# col1, col2 = st.columns(2)
# with col1:
#     any_healthcare = st.radio("Has Healthcare Coverage", ["No", "Yes"], key="healthcare")
# with col2:
#     no_doc_cost = st.radio("Couldn't See Doctor Due to Cost", ["No", "Yes"], key="nodoccost")

# st.markdown("#### 👤 Demographic Information")
# col1, col2, col3 = st.columns(3)
# with col1:
#     sex = st.radio("Sex", ["Male", "Female"], key="sex")
# with col2:
#     age = st.slider("Age", 0, 100, 30, key="age")
# with col3:
#     education = st.selectbox(
#         "Education Level", 
#         [
#             "Never attended school",
#             "Elementary school",
#             "Some high school",
#             "High school graduate",
#             "Some college",
#             "College graduate"
#         ],
#         key="education"
#     )

# income = st.selectbox(
#     "Annual Household Income", 
#     [
#         "Less than $10,000",
#         "$10,000-$15,000",
#         "$15,000-$20,000",
#         "$20,000-$25,000",
#         "$25,000-$35,000",
#         "$35,000-$50,000",
#         "$50,000-$75,000",
#         "$75,000 or more"
#     ],
#     key="income"
# )

# # Convert all inputs to model format - EXACT ORDER AS SPECIFIED
# def prepare_inputs():
#     # Map categorical variables to numerical
#     gen_hlth_map = {"Excellent":1, "Very Good":2, "Good":3, "Fair":4, "Poor":5}
#     education_map = {
#         "Never attended school": 1,
#         "Elementary school": 2,
#         "Some high school": 3,
#         "High school graduate": 4,
#         "Some college": 5,
#         "College graduate": 6
#     }
#     income_map = {
#         "Less than $10,000": 1,
#         "$10,000-$15,000": 2,
#         "$15,000-$20,000": 3,
#         "$20,000-$25,000": 4,
#         "$25,000-$35,000": 5,
#         "$35,000-$50,000": 6,
#         "$50,000-$75,000": 7,
#         "$75,000 or more": 8
#     }
    
#     # Create features in EXACTLY this order:
#     features = [
#         1 if high_bp == "Yes" else 0,        # HighBP (index 0)
#         1 if high_chol == "Yes" else 0,      # HighChol (1)
#         1 if chol_check == "Yes" else 0,     # CholCheck (2)
#         bmi,                                 # BMI (3)
#         1 if smoker == "Yes" else 0,         # Smoker (4)
#         1 if stroke == "Yes" else 0,         # Stroke (5)
#         1 if heart_disease == "Yes" else 0,  # HeartDiseaseorAttack (6)
#         1 if phys_activity == "Yes" else 0,  # PhysActivity (7)
#         1 if fruits == "Yes" else 0,         # Fruits (8)
#         1 if veggies == "Yes" else 0,        # Veggies (9)
#         1 if hvy_alcohol == "Yes" else 0,    # HvyAlcoholConsump (10)
#         1 if any_healthcare == "Yes" else 0, # AnyHealthcare (11)
#         1 if no_doc_cost == "Yes" else 0,    # NoDocbcCost (12)
#         gen_hlth_map[gen_hlth],              # GenHlth (13)
#         ment_hlth,                           # MentHlth (14)
#         phys_hlth,                           # PhysHlth (15)
#         1 if diff_walk == "Yes" else 0,      # DiffWalk (16)
#         1 if sex == "Female" else 0,         # Sex (17)
#         age,                                 # Age (18)
#         education_map[education],             # Education (19)
#         income_map[income]                   # Income (20)
#     ]
    
#     return np.array(features).reshape(1, -1)

# # Prediction button
# st.markdown("---")
# if st.button('Assess Diabetes Risk', key='predict_button'):
#     try:
#         input_data = prepare_inputs()
#         prediction = model.predict(input_data)
        
#         if hasattr(model, "predict_proba"):
#             prediction_proba = model.predict_proba(input_data)
#             # Display results
#             if prediction[0] == 0:
#                 st.success("🎉 **Low Risk**: You're unlikely to have diabetes.")
#             elif prediction[0] == 1:
#                 st.warning("⚠️ **Borderline Risk**: You may be at risk for diabetes.")
#             else:
#                 st.error("🛑 **High Risk**: You may have diabetes.")
            
#             # Show probabilities
#             st.markdown(f"**Probability Breakdown:**")
#             st.markdown(f"- No Diabetes: {prediction_proba[0][0]*100:.1f}%")
#             st.markdown(f"- Prediabetes/Borderline: {prediction_proba[0][1]*100:.1f}%")
#             st.markdown(f"- Diabetes: {prediction_proba[0][2]*100:.1f}%")
#         else:
#             st.success(f"Prediction result: {prediction[0]}")
            
#     except Exception as e:
#         st.error(f"🔴 Prediction error: {e}")

# # Footer
# st.markdown("---")
# st.markdown("### ℹ️ About This Tool")
# st.markdown("""
# This assessment predicts diabetes risk based on CDC behavioral risk factors. 
# It's not a substitute for professional medical advice.
# """)
# st.markdown("---")
# st.markdown("📊 **Model Information**: Random Forest Classifier trained on diabetes risk factors")
