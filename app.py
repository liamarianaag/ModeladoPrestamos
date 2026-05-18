import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Préstamos - ISIL", layout="centered")
st.title("🚀 Sistema de Evaluación de Préstamos")

# Requisito: Enlace al Colab
st.markdown("[🔗 Ver cuaderno de código en Google Colab](https://colab.research.google.com/drive/1jV7LpI9J6R6_P9z3G_S7q1N_G7U_Z0vS)") # Asegúrate de poner tu link real

# 2. Carga del modelo
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # 3. Formulario de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Ingreso del Solicitante ($)", min_value=0, value=5000)
        co_income = st.number_input("Ingreso del Co-solicitante ($)", min_value=0, value=0)
        loan_amount = st.number_input("Monto del Préstamo ($)", min_value=0, value=150)
        
    with col2:
        term = st.number_input("Plazo del Préstamo (Meses)", min_value=1, value=360)
        credit = st.selectbox("Historial Crediticio", options=[(1.0, "Bueno"), (0.0, "Malo")], format_func=lambda x: x[1])
        area = st.selectbox("Área de la Propiedad", options=[(0.0, "Rural"), (1.0, "Semiurbana"), (2.0, "Urbana")], format_func=lambda x: x[1])

    st.markdown("---")

    # 4. Botón y Predicción
    if st.button("Evaluar Préstamo"):
        # Cálculo de la cuota mensual
        cuota = float(loan_amount) / float(term)
        
        # NOMBRES EXACTOS DE COLUMNAS SEGÚN TU COLAB
        columnas = [
            'ApplicantIncome', 
            'CoapplicantIncome', 
            'LoanAmount', 
            'Loan_Amount_Term', 
            'Credit_History', 
            'Property_Area', 
            'Cuota_Mensual'
        ]
        
        # Creamos el DataFrame con los nombres correctos
        datos_entrada = pd.DataFrame([[
            float(income), 
            float(co_income), 
            float(loan_amount), 
            float(term), 
            float(credit[0]), 
            float(area[0]), 
            float(cuota)
        ]], columns=columnas)
        
        # Realizar la predicción
        prediccion = model.predict(datos_entrada)
        
        if prediccion[0] == 1:
            st.success("✅ **¡PRÉSTAMO APROBADO!**")
            st.balloons()
        else:
            st.error("❌ **PRÉSTAMO RECHAZADO.**")
            
else:
    st.error("⚠️ No se encontró el modelo en 'modelos/modelo_prestamos.pkl'.")
