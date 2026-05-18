import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la página
st.set_page_config(page_title="Predicción de Préstamos - ISIL", layout="centered")
st.title("🚀 Sistema de Evaluación de Préstamos")
st.write("Esta aplicación predice si un préstamo será aprobado basándose en el perfil del cliente.")

# Requisito: Enlace al Colab
st.markdown("[🔗 Ver cuaderno de código en Google Colab](https://colab.research.google.com/drive/TU_LINK_DE_COLAB_AQUI)")

# 2. Carga del modelo
# El archivo debe estar en la carpeta 'modelos' en GitHub
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # 3. Interfaz de usuario (Inputs)
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

    # 4. Lógica de Predicción
    if st.button("Evaluar Préstamo"):
        # Calculamos la variable que creaste en el segundo modelo
        cuota = loan_amount / term
        
        # DEFINICIÓN DE COLUMNAS (Orden exacto de tu X_train)
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
            float(income), 
            float(co_income), 
            float(loan_amount), 
            float(term), 
            float(credit[0]), 
            float(area[0]), 
            float(cuota)
        ]], columns=columnas)
        
        # Ejecutar predicción
        prediccion = model.predict(datos_entrada)
        
        if prediccion[0] == 1:
            st.success("✅ **¡PRÉSTAMO APROBADO!**")
            st.balloons()
        else:
            st.error("❌ **PRÉSTAMO RECHAZADO.** El perfil presenta un alto riesgo de impago.")
            
else:
    st.error("⚠️ No se encontró el modelo en 'modelos/modelo_prestamos.pkl'. Verifica tu repositorio de GitHub.")
