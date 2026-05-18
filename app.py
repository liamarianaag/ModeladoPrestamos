import streamlit as st
import pandas as pd
import joblib
import os

# 1. Configuración de la página
st.set_page_config(page_title="PA2 - Evaluación de Préstamos", layout="centered")

# --- INFORMACIÓN DEL ESTUDIANTE ---
st.sidebar.header("Datos de la Alumna")
st.sidebar.write("**Nombre:** Lia Mariana Aguilar Gonzales")
st.sidebar.write("**Código ISIL:** [Ingresa aquí tu código]") # Reemplaza con tu código real

# 2. Título y Recursos
st.title("🚀 Sistema de Evaluación de Préstamos")
st.write("Aplicación desarrollada para la predicción de riesgo crediticio basada en modelos de Machine Learning.")

st.markdown("### 🔗 Recursos del Proyecto")
# Enlace que me pasaste configurado como lector
st.info("📂 [Ver cuaderno de código en Google Colab (Lector)](https://colab.research.google.com/drive/1mrVzouL97p84LYqX6FKSUO78JU-3QTsY?usp=sharing)")

st.markdown("---")

# 3. Carga del modelo
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # 4. Formulario de entrada
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Ingreso del Solicitante ($)", min_value=0, value=2500)
        co_income = st.number_input("Ingreso del Co-solicitante ($)", min_value=0, value=1200)
        loan_amount = st.number_input("Monto del Préstamo ($)", min_value=0, value=15000)
    with col2:
        term = st.number_input("Plazo del Préstamo (Meses)", min_value=1, value=120)
        credit = st.selectbox("Historial Crediticio", options=[(1.0, "Bueno"), (0.0, "Malo")], format_func=lambda x: x[1])
        area = st.selectbox("Área de la Propiedad", options=[(0, "Rural"), (1, "Semiurbana"), (2, "Urbana")], format_func=lambda x: x[1])

    # 5. Botón y Predicción
    if st.button("Evaluar Solicitud"):
        # Variable calculada (Ingeniería de características)
        cuota = float(loan_amount) / float(term)
        
        # Nombres de columnas idénticos a tu entrenamiento en Colab
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
            float(income), float(co_income), float(loan_amount), 
            float(term), float(credit[0]), float(area[0]), float(cuota)
        ]], columns=columnas)
        
        # Ejecutar el modelo XGBoost
        prediccion = model.predict(datos_entrada)
        
        if prediccion[0] == 1:
            st.success("✅ **¡PRÉSTAMO APROBADO!**")
            st.balloons()
        else:
            st.error("❌ **PRÉSTAMO RECHAZADO.** El perfil no cumple con los parámetros de seguridad financiera.")
else:
    st.error("⚠️ Error: El archivo 'modelo_prestamos.pkl' no se encuentra en la carpeta 'modelos/'.")
