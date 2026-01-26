import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Obesity Risk Predictor", layout="wide")

# Load the saved pipeline
import os
@st.cache_resource
def load_model():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model_pipeline.pkl')
    return joblib.load(model_path)

model = load_model()

# Header
st.title("🏥 Obesity Risk Assessment System")
st.markdown("""
This tool uses a Machine Learning model to assist medical professionals in identifying obesity risk levels 
based on physical data and lifestyle habits.
""")

st.divider()

# Creating columns for the layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Personal Data")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
    family_history = st.selectbox("Family History of Overweight?", ["yes", "no"])

with col2:
    st.header("🥗 Eating Habits")
    favc = st.selectbox("Frequent consumption of high-calorie food?", ["yes", "no"])
    fcvc = st.slider("Frequency of vegetable consumption (1-3)", 1, 3, 2)
    ncp = st.slider("Number of main meals (1-4)", 1, 4, 3)
    caec = st.selectbox("Consumption of food between meals", ["no", "Sometimes", "Frequently", "Always"])
    ch2o = st.slider("Daily water consumption (1-3)", 1, 3, 2)
    scc = st.selectbox("Daily calorie monitoring?", ["yes", "no"])

with col3:
    st.header("🏃 Lifestyle & Transport")
    faf = st.slider("Physical activity frequency (0-3)", 0, 3, 1)
    tue = st.slider("Time using electronic devices (0-2)", 0, 2, 1)
    smoke = st.selectbox("Smoker?", ["yes", "no"])
    calc = st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Main mode of transport", 
                          ["Public_Transportation", "Automobile", "Motorbike", "Bike", "Walking"])

# Prediction Logic
st.divider()
if st.button("🔍 Predict Health Status", type="primary"):
    # Create a dictionary with the inputs
    input_data = {
        'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
        'family_history': family_history, 'FAVC': favc, 'FCVC': fcvc,
        'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o,
        'SCC': scc, 'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Get prediction
    prediction = model.predict(input_df)[0]
    
    # Display Result
    st.subheader("Diagnostic Result:")
    # Formatting the result for better reading
    result_display = prediction.replace("_", " ")
    
    if "Obesity" in prediction:
        st.error(f"Prediction: {result_display}")
    elif "Overweight" in prediction:
        st.warning(f"Prediction: {result_display}")
    else:
        st.success(f"Prediction: {result_display}")