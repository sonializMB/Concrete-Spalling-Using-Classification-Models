import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt


#CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Predicción de Spalling",
    layout="wide"
)

#ESTILOS
st.markdown("""
    <style>
    .titulo {
        font-size: 2rem;
        font-weight: 700;
    }
    .subtitulo {
        font-size: 1rem;
        opacity: 0.8;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo">Predicción de Spalling en Concreto</div>', unsafe_allow_html=True)

#CARGAR MODELOS
modelos = {
    "Regresión Logística": joblib.load("modelos_guardados/Regresion_Logística.pkl"),
    "KNN": joblib.load("modelos_guardados/KNN.pkl"),
    "Naive Bayes": joblib.load("modelos_guardados/Naive_Bayes.pkl"),
    "SVM": joblib.load("modelos_guardados/SVM.pkl"),
    "Árbol de Decisión": joblib.load("modelos_guardados/Árbol_de_Decision.pkl"),
    "Perceptrón": joblib.load("modelos_guardados/Perceptron.pkl")
}

# SIGNFICADO DE LAS VARIABLES DE ENTRADA
with st.expander("**Significado de datos de entrada**"):
    st.markdown("""
    **Variables de entrada:**
    - W/B: Relación agua/cemento
    - CA/B: Relación agregado grueso/cemento
    - FA/B: Relación agregado fino/cemento
    - F/B: Relación aditivo/cemento
    - GGBS/B: Relación escoria de alto horno/cemento
    - SF/B: Relación humo de sílice/cemento
    - Sa: Área superficial del agregado fino (m²/kg)
    - FiberType: Tipo de fibra utilizada (categoría)
    - FiberVol: Volumen de fibra (%)
    - HrRate: Tasa de curado (hr⁻¹)
    - Tmax: Temperatura máxima durante el curado (°C)
    - Time: Tiempo de exposición al ambiente agresivo (días)
    - Diam: Diámetro máximo del agregado grueso (mm)
    - SpecVol: Volumen específico del agregado grueso (cm³/g)
    - Age: Edad del concreto al momento de la evaluación (días)
    - CS_28d: Resistencia a compresión a los 28 días (MPa)
    """, unsafe_allow_html=True)


# FORMULARIO DE INGRESO DE DATOS
st.markdown('<div class="section-title">Ingreso de variables</div>', unsafe_allow_html=True)

with st.form("form_spalling"):
    col1, col2, col3, col4 = st.columns(4)

    with col1: #Columna 1
        w_b = st.number_input("W/B", value=0.30, format="%.2f")
        ca_b = st.number_input("CA/B", value=1.50, format="%.2f")
        fa_b = st.number_input("FA/B", value=1.20, format="%.2f")
        f_b = st.number_input("F/B", value=0.00, format="%.2f")

    with col2: #Columna 2
        ggbs_b = st.number_input("GGBS/B", value=0.25, format="%.2f")
        sf_b = st.number_input("SF/B", value=0.00, format="%.2f")
        sa = st.number_input("Sa", value=7.0, format="%.2f")
        fiber_type = st.selectbox(
            "FiberType",
            options=[
                '0', 'BF', 'CF', 'CF+PP', 'CF+SF', 'CL+PP', 'FF', 'JF', 'LLDPE',
                'NY', 'NY+PP', 'PA', 'PE', 'PET', 'PP', 'PP+NY', 'PP+PE', 'PP+PP',
                'PVA', 'PVA+PP', 'SF', 'SF+BF', 'SF+FF', 'SF+PE', 'SF+PP',
                'SF+PVA', 'SF+SF+PP', 'SF+SF+PVA', 'UHMWPE'
            ]
        )

    with col3: #columna 3
        fiber_vol = st.number_input("FiberVol", value=0.0, format="%.2f")
        hr_rate = st.number_input("HrRate", value=5.0, format="%.2f")
        tmax = st.number_input("Tmax", value=600.0, format="%.2f")
        time = st.number_input("Time", value=60.0, format="%.2f")


    with col4: #Columna 4
        diam = st.number_input("Diam", value=100.0, format="%.2f")
        spec_vol = st.number_input("SpecVol", value=1.8, format="%.2f")
        age = st.number_input("Age", value=28.0, format="%.2f")
        cs_28d = st.number_input("CS_28d", value=40.0, format="%.2f")

    submitted = st.form_submit_button("Predecir") #Botón para enviar el formulario y realizar la predicción

#DATAFRAME DE ENTRADA
entrada = pd.DataFrame([{
    "W/B": w_b,
    "CA/B": ca_b,
    "FA/B": fa_b,
    "F/B": f_b,
    "GGBS/B": ggbs_b,
    "SF/B": sf_b,
    "Sa": sa,
    "FiberType": fiber_type,
    "FiberVol": fiber_vol,
    "HrRate": hr_rate,
    "Tmax": tmax,
    "Time": time,
    "Diam": diam,
    "SpecVol": spec_vol,
    "Age": age,
    "CS_28d": cs_28d
}])
#Mostrar el dataframe de entrada para verificar los datos ingresados por el usuario
st.dataframe(entrada, width="stretch") 


# PREDICCIÓN
if submitted:
    resultados = []

    for nombre, modelo in modelos.items():
        pred = modelo.predict(entrada)[0]

        if hasattr(modelo, "predict_proba"):
            prob = modelo.predict_proba(entrada)[0][1]
        elif hasattr(modelo, "decision_function"):
            score = modelo.decision_function(entrada)[0]
            prob = score
        else:
            prob = np.nan

        resultados.append({
            "Modelo": nombre,
            "Predicción": "Spalling" if pred == 1 else "No Spalling",
            "Probabilidad Spalling": f"{prob*100:.4f}%" if not np.isnan(prob) else "N/A"
        })

    df_resultados = pd.DataFrame(resultados)

    #Mostrar resultados del modelo elegido(SVM)
    modelo_final = modelos["SVM"]
    probs_final = modelo_final.predict_proba(entrada)[0]
    pred_final = modelo_final.predict(entrada)[0]

    labels = ["No Spalling", "Spalling"]
    st.subheader("Predicción final del modelo seleccionado")
    #st.markdown(probs_final, unsafe_allow_html=True)
    if pred_final == 1:
        st.error(f"Resultado final: Spalling ({probs_final[1]*100:.4f}%)")
    else:
        st.success(f"Resultado final: No Spalling ({probs_final[0]*100:.4f}%)")

    fig, ax = plt.subplots(figsize=(1, 1))
    #GRafica de pastel para mostrar probabilidades del modelo SVM
    wedges, texts, autotexts=ax.pie(
        probs_final, #probabilidades de cada clase
        labels=labels, # etiquetas de las clases
        autopct="%1.1f%%", # formato de porcentaje
        startangle=90
    )
    
    for text in texts:
        text.set_fontsize(5) #tamaño de fuente para las etiquetas de las clases
    
    for autotext in autotexts:
        autotext.set_fontsize(4)#tamaño de fuente para los porcentajes
    
    ax.set_title("Predicción del modelo final (SVM)",fontsize=8) #título de la gráfica
    ax.axis("equal")#Para que el gráfico sea circular
    
    st.pyplot(fig)

    
    #Mostrar resultados de todos los modelos
    with st.expander("**Resultados de los distintos modelos**"):
        st.markdown("""
        **Resultados de predicción:**
        - Modelo: Nombre del modelo de clasificación utilizado.
        - Predicción: Resultado de la predicción (Spalling o No Spalling).
        - Probabilidad: Probabilidad asociada a la clase "Spalling".
        """, unsafe_allow_html=True)
        st.dataframe(df_resultados, width="stretch")