import streamlit as st

# Set page config (must be first)
st.set_page_config(page_title="Assessment Results", layout="wide")

# Check if results exist
if 'results' not in st.session_state:
    st.error("No assessment results found. Please complete the assessment form first.")
    
    # Create a button to go back to the form
    if st.button("Go to Assessment Form"):
        st.switch_page("pages/input_form.py")
    
    # Stop execution if no results
    st.stop()

# Get results from session state
results = st.session_state['results']

# Display results
st.title("Diabetes Risk Assessment Results")

# Display the prediction
prediction = results['prediction']
proba = results['prediction_prob']
risk_level = "high" if prediction == 1 else "low"
confidence = proba if prediction == 1 else (1 - proba)

st.subheader(f"Your diabetes risk is: {risk_level.upper()}")
st.progress(int(confidence * 100))
st.caption(f"Confidence: {confidence:.1%}")

# Show recommendations
if risk_level == "high":
    st.warning("""
    - Consult with a healthcare provider
    - Consider lifestyle changes
    - Monitor blood sugar regularly
    """)
else:
    st.success("""
    - Maintain healthy habits
    - Get regular check-ups
    - Continue monitoring
    """)

# Add button to return to form
if st.button("Perform New Assessment"):
    st.switch_page("pages/input_form.py")