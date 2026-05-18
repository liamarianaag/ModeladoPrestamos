import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración y Título
st.set_page_config(page_title="Predicción de Préstamos - ISIL", layout="centered")
st.title("🚀 Sistema de Evaluación de Préstamos")

# 2. Cargar el modelo
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # 3. Formulario con DATOS REALISTAS (como tú dijiste)
    col1, col2 = st.columns(2)
    
    with col1:
        # Puse valores por defecto más lógicos
        income = st.number_input("Tu Ingreso Mensual ($)", min_value=0, value=2500)
        co_income = st.number_input("Ingreso de Co-solicitante ($)", min_value=0, value=1200)
        loan_amount = st.number_input("Monto que solicitas ($)", min_value=0, value=15000)
        
    with col2:
        term = st.number_input("Plazo en meses", min_value=1, value=120)
        # 1 es Bueno, 0 es Malo
        credit = st.selectbox("¿Cómo es tu historial?", options=[(1, "Bueno"), (0, "Malo")], format_func=lambda x: x[1])
        # 0: Rural, 1: Semiurbana, 2: Urbana
        area = st.selectbox("Zona de la propiedad", options=[(0, "Rural"), (1, "Semiurbana"), (2, "Urbana")], format_func=lambda x: x[1])

    if st.button("Evaluar mi solicitud"):
        # Calculamos la cuota
        cuota = loan_amount / term
        
        # ORDEN DE COLUMNAS CRÍTICO PARA XGBOOST
        # He verificado este orden con tu archivo .ipynb
        columnas_entrenamiento = [
            'ApplicantIncome', 
            'CoapplicantIncome', 
            'LoanAmount', 
            'Loan_Amount_Term', 
            'Credit_History', 
            'Property_Area', 
            'Cuota_Mensual'
        ]
        
        # Creamos los datos de entrada
        datos_entrada = pd.DataFrame([[
            float(income), 
            float(co_income), 
            float(loan_amount), 
            float(term), 
            float(credit[0]), 
            float(area[0]), 
            float(cuota)
        ]], columns=columnas_entrenamiento)
        
        # Predicción
        prediccion = model.predict(datos_entrada)
        
        if prediccion[0] == 1:
            st.success("✅ ¡PRÉSTAMO APROBADO!")
            st.balloons()
        else:
            st.error("❌ PRÉSTAMO RECHAZADO por riesgo financiero.")
else:
    st.error("Archivo de modelo no encontrado.")
