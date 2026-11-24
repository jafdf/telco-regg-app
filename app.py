import streamlit as st
import pickle

# Cargar el modelo y el DictVectorizer
with open('churn-model.pck', 'rb') as f:
    dv, model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Churn de Clientes")

# Formulario para introducir datos del cliente
st.header("Introduce los datos del cliente:")
st.markdown("*(Se solicitan únicamente los 11 atributos más importantes para el cálculo)*")

# --- Los 11 atributos más predictivos ---
contract = st.selectbox("Tipo de contrato", ["month-to-month", "one_year", "two_year"])
tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, step=1)
monthly_charges = st.number_input("Cargos mensuales", min_value=0.0, max_value=500.0, step=0.1)
total_charges = st.number_input("Cargos totales", min_value=0.0, max_value=10000.0, step=0.1)
internet_service = st.selectbox("Tipo de Internet", ["dsl", "fiber_optic", "no"])
payment_method = st.selectbox("Método de pago", ["bank_transfer_(automatic)", "credit_card_(automatic)", "electronic_check", "mailed_check"])
online_security = st.selectbox("Seguridad online", ["no", "yes", "no_internet_service"])
tech_support = st.selectbox("Soporte técnico", ["no", "yes", "no_internet_service"])
paperless_billing = st.selectbox("Facturación sin papel", ["no", "yes"])
online_backup = st.selectbox("Copia de seguridad online", ["no", "yes", "no_internet_service"])
device_protection = st.selectbox("Protección de dispositivos", ["no", "yes", "no_internet_service"])

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos ingresados y rellenar el resto con valores base (moda)
    client_data = {
        # Atributos principales ingresados en el formulario
        "contract": contract,
        "tenure": tenure,
        "monthlycharges": monthly_charges,
        "totalcharges": total_charges,
        "internetservice": internet_service,
        "paymentmethod": payment_method,
        "onlinesecurity": online_security,
        "techsupport": tech_support,
        "paperlessbilling": paperless_billing,
        "onlinebackup": online_backup,
        "deviceprotection": device_protection,
        
        # Atributos secundarios ocultos rellenados por defecto para mantener la estructura original
        "dependents": "no",
        "gender": "female",
        "multiplelines": "no",
        "partner": "no",
        "phoneservice": "yes",
        "seniorcitizen": "no",
        "streamingmovies": "no",
        "streamingtv": "no"
    }

    # Transformar los datos del cliente
    X_client = dv.transform([client_data])

    # Realizar la predicción
    y_pred_proba = model.predict_proba(X_client)[0][1]  # Probabilidad de churn

    # Mostrar resultado
    st.subheader("Resultado:")
    if y_pred_proba > 0.5:
        st.error(f"El cliente tiene una alta probabilidad de churn: {y_pred_proba:.2f}")
    else:
        st.success(f"El cliente tiene una baja probabilidad de churn: {y_pred_proba:.2f}")
