# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregando o modelo treinado
loaded_model = load_model('../soil_classifier_model.h5')

# Função para pré-processar a imagem
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizando os pixels para o intervalo [0, 1]
    return img_array

# Função para fazer a predição com probabilidades para ambas as classes
def predict_soil_type_with_prob(img_path):
    img_array = preprocess_image(img_path)
    predictions = loaded_model.predict(img_array)
    
    # Obtendo as probabilidades associadas às classes "black" e "red"
    black_probability = predictions[0][0]
    red_probability = 1 - black_probability
    
    return black_probability, red_probability

# Interface Streamlit
st.title("Predição de Solo")
uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    # Prever solo
    black_probability, red_probability = predict_soil_type_with_prob(uploaded_file)
    
    # Comparando as probabilidades e atribuindo a classe com maior probabilidade
    if black_probability > red_probability:
        predicted_class = "black"
        higher_probability = black_probability
    else:
        predicted_class = "red"
        higher_probability = red_probability

    # Exibir resultados na interface
    st.image(uploaded_file, caption="Imagem de Solo", use_column_width=True)
    st.write(f"A classe do solo é: {predicted_class}")
    st.write(f"Probabilidade de ser solo preto: {black_probability * 100:.2f}%")
    st.write(f"Probabilidade de ser solo vermelho: {red_probability * 100:.2f}%")
