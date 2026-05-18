import streamlit as st
import pandas as pd
import joblib
import os

# Configuración de la página
st.title("🚀 Predicción de Aprobación de Préstamos")
st.write("Ingresa los datos del cliente para evaluar el riesgo de crédito.")

# Enlace a tu Colab (Requisito del punto 10)
st.markdown("[🔗 Ver cuaderno de código en Google Colab](TU_LINK_DE_COLAB_AQUI)")

# 1. Cargar el modelo
# Buscamos el modelo dentro de la carpeta 'modelos'
modelo_path = os.path.join('modelos', 'modelo_prestamos.pkl')

if os.path.exists(modelo_path):
    model = joblib.load(modelo_path)
    
    # 2. Crear el formulario de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Ingreso del Solicitante ($)", min_value=0)
        co_income = st.number_input("Ingreso del Co-solicitante ($)", min_value=0)
        loan_amount = st.number_input("Monto del Préstamo ($)", min_value=0)
        
    with col2:
        term = st.number_input("Plazo del Préstamo (Meses)", min_value=1, value=360)
        credit = st.selectbox("Historial Crediticio", options=[(1, "Bueno"), (0, "Malo")], format_func=lambda x: x[1])
        area = st.selectbox("Área de la Propiedad", options=[(0, "Rural"), (1, "Semiurbana"), (2, "Urbana")], format_func=lambda x: x[1])

    # 3. Botón de Predicción
    if st.button("Evaluar Préstamo"):
        # Calculamos la variable que inventamos
        cuota = loan_amount / term
        
        # Creamos el DataFrame para el modelo (debe tener el mismo orden que en el entrenamiento)
        datos_entrada = pd.DataFrame([[
            income, co_income, loan_amount, term, credit[0], area[0], cuota
        ]], columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Cuota_Mensual'])
        
        # Predicción
        prediccion = model.predict(datos_entrada)
        
        if prediccion[0] == 1:
            st.success("✅ ¡Préstamo Aprobado!")
        else:
            st.error("❌ Préstamo Rechazado (Alto Riesgo)")
else:
    st.error("No se encontró el modelo en la carpeta 'modelos/'. Por favor, verifica tu GitHub.")
