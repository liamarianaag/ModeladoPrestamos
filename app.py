import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Préstamos - ISIL", layout="centered")
st.title("🚀 Sistema de Evaluación de Préstamos")
st.write("Ingresa los datos para determinar si el préstamo debe ser aprobado o rechazado.")

# Enlace al Colab (Requisito del proyecto)
st.markdown("[🔗 Ver cuaderno de código en Google Colab](https://colab.research.google.com/drive/TU_LINK_AQUI)")

# 2. Cargar el modelo guardado
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # 3. Formulario de entrada de datos
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("Ingreso del Solicitante ($)", min_value=0, value=5000)
            co_income = st.number_input("Ingreso del Co-solicitante ($)", min_value=0, value=0)
            loan_amount = st.number_input("Monto del Préstamo ($)", min_value=0, value=150)
            
        with col2:
            term = st.number_input("Plazo del Préstamo (Meses)", min_value=1, value=360)
            credit = st.selectbox("Historial Crediticio", options=[(1, "Bueno"), (0, "Malo")], format_func=lambda x: x[1])
            area = st.selectbox("Área de la Propiedad", options=[(0, "Rural"), (1, "Semiurbana"), (2, "Urbana")], format_func=lambda x: x[1])

    st.markdown("---")

    # 4. Botón y Lógica de Predicción
    if st.button("Evaluar Préstamo"):
        # Cálculo de la nueva variable (Ingeniería de Características)
        cuota = loan_amount / term
        
        # Nombres de columnas EXACTOS como en el entrenamiento (PA2_MachineLearning_AguilarLia.ipynb)
        columnas = [
            'ApplicantIncome', 
            'CoapplicantIncome', 
            'LoanAmount', 
            'Loan_Amount_Term', 
            'Credit_History', 
            'Property_Area', 
            'Cuota_Mensual'
        ]
        
        # Crear DataFrame para el modelo
        datos_entrada = pd.DataFrame([[
            income, 
            co_income, 
            loan_amount, 
            term, 
            credit[0], 
            area[0], 
            cuota
        ]], columns=columnas)
        
        # Realizar la predicción
        prediccion = model.predict(datos_entrada)
        
        # Mostrar resultado
        if prediccion[0] == 1:
            st.success("✅ **¡PRÉSTAMO APROBADO!** El cliente cumple con el perfil de bajo riesgo.")
            st.balloons()
        else:
            st.error("❌ **PRÉSTAMO RECHAZADO.** El sistema detecta un alto riesgo de impago.")
            
else:
    st.error("⚠️ Error crítico: No se encontró el archivo 'modelo_prestamos.pkl' en la carpeta 'modelos/'.")
