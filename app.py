import streamlit as st
import pandas as pd
import pickle
from pycaret.classification import predict_model
import tempfile
import shutil

# Definir el path del modelo
path = "/code/Python/Corte_2/Quiz_2_2/Punto_4/"

# Cargar el modelo preentrenado
with open(path + 'models/ridge_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

# Título de la aplicación
st.title("API de Predicción Estudiantes")

# Botón para subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

# Función para limpiar y convertir columnas numéricas
def clean_and_convert(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(',', '.'), errors='coerce')

# Botón para predecir
if st.button("Predecir"):
    if uploaded_file is not None:
        try:
            # Crear un archivo temporal para manejar el archivo subido
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                tmp_path = temp_file.name

            # Leer el archivo Excel o CSV usando pandas
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(tmp_path)
            else:
                df = pd.read_excel(tmp_path)

            # Limpiar y convertir las columnas numéricas deseadas
            for column in ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']:
                clean_and_convert(df, column)

            # Realizar predicción usando el modelo cargado
            predictions = predict_model(modelo, data=df)

            # Redondear las predicciones a 4 decimales y agregarlas al DataFrame original
            df['Predicciones'] = predictions["prediction_label"].round(4)

            # Mostrar el DataFrame con las predicciones
            st.write("Predicciones agregadas correctamente al archivo:")
            st.write(df)

            # Guardar el archivo Excel o CSV con las predicciones incluidas
            if uploaded_file.name.endswith(".csv"):
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Descargar archivo con predicciones",
                                   data=csv_data,
                                   file_name="predicciones_con_resultados.csv",
                                   mime="text/csv")
            else:
                excel_data = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                df.to_excel(excel_data.name, index=False)
                st.download_button(label="Descargar archivo con predicciones",
                                   data=excel_data.read(),
                                   file_name="predicciones_con_resultados.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Por favor, cargue un archivo válido.")

# Botón para reiniciar la página
if st.button("Reiniciar"):
    st.rerun()
