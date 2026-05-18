import streamlit as st
import pandas as pd
import joblib
import os

# Configuración básica
st.set_page_config(page_title="Predicción de Préstamos", layout="centered")
st.title("🚀 Sistema de Evaluación de Préstamos")

# Carga del modelo
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # Formulario
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Ingreso del Solicitante", min_value=0, value=5000)
        co_income = st.number_input("Ingreso del Co-solicitante", min_value=0, value=0)
        loan_amount = st.number_input("Monto del Préstamo", min_value=0, value=150)
    with col2:
        term = st.number_input("Plazo (Meses)", min_value=1, value=360)
        credit = st.selectbox("Historial Crediticio", options=[(1.0, "Bueno"), (0.0, "Malo")], format_func=lambda x: x[1])
        area = st.selectbox("Área de la Propiedad", options=[(0, "Rural"), (1, "Semiurbana"), (2, "Urbana")], format_func=lambda x: x[1])

    if st.button("Evaluar"):
        cuota = float(loan_amount) / float(term)
        
        # NOMBRES EXACTOS SEGÚN TU COLAB (XGBOOST ES MUY ESTRICTO)
        columnas = [
            'ApplicantIncome', 
            'CoapplicantIncome', 
            'LoanAmount', 
            'Loan_Amount_Term', 
            'Credit_History', 
            'Property_Area', 
            'Cuota_Mensual'
        ]
        
        datos_entrada = pd.DataFrame([[
            float(income), 
            float(co_income), 
            float(loan_amount), 
            float(term), 
            float(credit[0]), 
            float(area[0]), 
            float(cuota)
        ]], columns=columnas)
        
        # Predicción
        prediccion = model.predict(datos_entrada)
        
        if prediccion[0] == 1:
            st.success("✅ ¡Préstamo Aprobado!")
            st.balloons()
        else:
            st.error("❌ Préstamo Rechazado")
else:
    st.error("No se encontró el archivo del modelo.")
