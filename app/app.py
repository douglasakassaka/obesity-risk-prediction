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

# Variable Descriptions
with st.expander("ℹ️ Variable Descriptions - Data Dictionary"):
    st.markdown("""
    ### 👤 Personal Data
    - **Gender**: Sexo do paciente (Female/Male)
    - **Age**: Idade do paciente em anos
    - **Height**: Altura do paciente em metros
    - **Weight**: Peso do paciente em quilogramas
    - **family_history**: Histórico familiar de sobrepeso (yes/no)
    
    ### 🥗 Eating Habits (Hábitos Alimentares)
    - **FAVC** (Frequent consumption of high-calorie food): Consumo frequente de alimentos com alto teor calórico (yes/no)
    - **FCVC** (Frequency of Consumption of Vegetables): Frequência de consumo de vegetais nas refeições (escala 1-3)
      - 1 = Nunca
      - 2 = Às vezes
      - 3 = Sempre
    - **NCP** (Number of main meals): Número de refeições principais por dia (1-4)
    - **CAEC** (Consumption of food between meals): Consumo de alimentos entre as refeições
      - no = Não
      - Sometimes = Às vezes
      - Frequently = Frequentemente
      - Always = Sempre
    - **CH2O** (Consumption of water daily): Consumo diário de água em litros (escala 1-3)
      - 1 = Menos de 1L
      - 2 = 1-2L
      - 3 = Mais de 2L
    - **SCC** (Calories consumption monitoring): Monitora o consumo de calorias diariamente? (yes/no)
    
    ### 🏃 Lifestyle & Transport (Estilo de Vida e Transporte)
    - **FAF** (Physical activity frequency): Frequência de atividade física por semana (escala 0-3)
      - 0 = Nenhuma
      - 1 = 1-2 dias
      - 2 = 2-4 dias
      - 3 = 4-5 dias
    - **TUE** (Time using technology devices): Tempo de uso de dispositivos eletrônicos por dia (escala 0-2)
      - 0 = 0-2 horas
      - 1 = 3-5 horas
      - 2 = Mais de 5 horas
    - **SMOKE**: Fumante? (yes/no)
    - **CALC** (Consumption of alcohol): Consumo de álcool
      - no = Não
      - Sometimes = Às vezes
      - Frequently = Frequentemente
      - Always = Sempre
    - **MTRANS** (Transportation used): Meio de transporte principal utilizado
      - Public_Transportation = Transporte público
      - Automobile = Automóvel
      - Motorbike = Motocicleta
      - Bike = Bicicleta
      - Walking = Caminhada
    
    ### 🎯 Target Variable (Variável Alvo)
    - **Obesity**: Nível de obesidade classificado em 7 categorias:
      - Insufficient_Weight = Peso insuficiente
      - Normal_Weight = Peso normal
      - Overweight_Level_I = Sobrepeso Nível I
      - Overweight_Level_II = Sobrepeso Nível II
      - Obesity_Type_I = Obesidade Tipo I
      - Obesity_Type_II = Obesidade Tipo II
      - Obesity_Type_III = Obesidade Tipo III
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