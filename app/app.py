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
if st.button("🔍 Predict Health Status", type="primary", use_container_width=True):
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
    
    # Calculate BMI
    bmi = weight / (height ** 2)
    
    # Display Results with enhanced visualization
    st.divider()
    st.header("📊 Health Assessment Results")
    
    # Create two columns for metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(label="📏 BMI Calculated", value=f"{bmi:.1f}")
    
    with metric_col2:
        st.metric(label="⚖️ Weight", value=f"{weight} kg")
    
    with metric_col3:
        st.metric(label="📐 Height", value=f"{height} m")
    
    st.divider()
    
    # Main result display with color coding
    result_display = prediction.replace("_", " ").title()
    
    # Define status levels and recommendations
    status_config = {
        "Insufficient Weight": {
            "emoji": "⚠️",
            "color": "blue",
            "message": "Peso Insuficiente",
            "recommendation": "• Consulte um nutricionista para plano alimentar adequado\n• Avalie possíveis deficiências nutricionais\n• Considere suplementação se necessário"
        },
        "Normal Weight": {
            "emoji": "✅",
            "color": "green",
            "message": "Peso Normal - Parabéns!",
            "recommendation": "• Mantenha hábitos alimentares saudáveis\n• Continue praticando atividades físicas regulares\n• Realize check-ups preventivos anuais"
        },
        "Overweight Level I": {
            "emoji": "⚡",
            "color": "orange",
            "message": "Sobrepeso Nível I",
            "recommendation": "• Inicie ou intensifique atividade física (150 min/semana)\n• Ajuste padrão alimentar reduzindo calorias\n• Acompanhamento nutricional é recomendado"
        },
        "Overweight Level Ii": {
            "emoji": "⚡",
            "color": "orange",
            "message": "Sobrepeso Nível II",
            "recommendation": "• Consulte médico e nutricionista urgentemente\n• Estabeleça meta de redução de peso gradual\n• Atividade física supervisionada é importante"
        },
        "Obesity Type I": {
            "emoji": "🔴",
            "color": "red",
            "message": "Obesidade Tipo I",
            "recommendation": "• Acompanhamento médico multiprofissional necessário\n• Avalie riscos cardiovasculares e metabólicos\n• Plano estruturado de perda de peso com metas\n• Considere apoio psicológico"
        },
        "Obesity Type Ii": {
            "emoji": "🔴",
            "color": "red",
            "message": "Obesidade Tipo II",
            "recommendation": "• Tratamento médico intensivo é essencial\n• Avaliação de comorbidades (diabetes, hipertensão)\n• Considere tratamento farmacológico\n• Suporte multidisciplinar completo"
        },
        "Obesity Type Iii": {
            "emoji": "🚨",
            "color": "red",
            "message": "Obesidade Tipo III (Mórbida)",
            "recommendation": "• Procure atendimento médico especializado IMEDIATAMENTE\n• Avaliação para cirurgia bariátrica pode ser necessária\n• Monitoramento rigoroso de comorbidades\n• Suporte psicológico e nutricional intensivo"
        }
    }
    
    # Get configuration for current prediction
    config = status_config.get(result_display, status_config["Normal Weight"])
    
    # Display result in colored container
    if config["color"] == "green":
        st.success(f"### {config['emoji']} {config['message']}")
    elif config["color"] == "blue":
        st.info(f"### {config['emoji']} {config['message']}")
    elif config["color"] == "orange":
        st.warning(f"### {config['emoji']} {config['message']}")
    else:  # red
        st.error(f"### {config['emoji']} {config['message']}")
    
    # Recommendations section
    st.subheader("💡 Recomendações Médicas")
    st.markdown(config["recommendation"])
    
    # Risk factors summary
    st.divider()
    st.subheader("📋 Resumo dos Fatores de Risco")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("**Fatores Positivos:**")
        positive_factors = []
        if faf >= 2:
            positive_factors.append("✓ Atividade física regular")
        if fcvc >= 2:
            positive_factors.append("✓ Bom consumo de vegetais")
        if ch2o >= 2:
            positive_factors.append("✓ Boa hidratação")
        if scc == "yes":
            positive_factors.append("✓ Monitora calorias")
        if smoke == "no":
            positive_factors.append("✓ Não fumante")
        if mtrans in ["Bike", "Walking"]:
            positive_factors.append("✓ Transporte ativo")
        
        if positive_factors:
            for factor in positive_factors:
                st.write(factor)
        else:
            st.write("Nenhum fator positivo identificado")
    
    with risk_col2:
        st.markdown("**Fatores de Atenção:**")
        risk_factors = []
        if faf == 0:
            risk_factors.append("⚠ Sedentarismo")
        if favc == "yes":
            risk_factors.append("⚠ Alto consumo calórico")
        if fcvc == 1:
            risk_factors.append("⚠ Baixo consumo de vegetais")
        if caec in ["Frequently", "Always"]:
            risk_factors.append("⚠ Belisca frequentemente")
        if ch2o == 1:
            risk_factors.append("⚠ Baixa hidratação")
        if tue == 2:
            risk_factors.append("⚠ Muito tempo em telas")
        if calc in ["Frequently", "Always"]:
            risk_factors.append("⚠ Alto consumo de álcool")
        if smoke == "yes":
            risk_factors.append("⚠ Fumante")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("Nenhum fator de risco identificado")
    
    # Footer note
    st.divider()
    st.info("ℹ️ **Nota:** Esta é uma avaliação automatizada para apoio à decisão médica. Sempre consulte um profissional de saúde para diagnóstico e tratamento adequados.")