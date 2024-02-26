import streamlit as st
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar modelo treinado
loaded_model = load_model('../soil_classifier_model.h5')

# Conectar ao Azure Blob Storage
azure_storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=solowisetest;AccountKey=2QfzgMQ7o0EC92Vdb71roEGBzlVyPigVjTVA5sJKJLoRQOIpyVQeVD81Xub1dAYX9v7nZi71H85D+AStF2L3Uw==;EndpointSuffix=core.windows.net"
container_name = "container1"
blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)

# Função para pré-processar a imagem
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar os pixels para o intervalo [0, 1]
    return img_array

# Função para fazer a predição com probabilidades para ambas as classes
def predict_soil_type_with_prob(img_path):
    img_array = preprocess_image(img_path)
    predictions = loaded_model.predict(img_array)
    black_probability = predictions[0][0]
    red_probability = 1 - black_probability
    return black_probability, red_probability

# Interface Streamlit
st.title("Predição de Solo")

# Permitir upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Upload da imagem para o Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=uploaded_file.name)
    blob_client.upload_blob(uploaded_file)

    # Prever solo
    black_probability, red_probability = predict_soil_type_with_prob(uploaded_file)
    if black_probability > red_probability:
        predicted_class = "Preto"
        higher_probability = black_probability
    else:
        predicted_class = "Vermelho"
        higher_probability = red_probability

    # Exibir resultados na interface
    st.image(uploaded_file, caption="Imagem de Solo", use_column_width=True)
    st.write(f"Classe do solo: {predicted_class}")
    st.write(f"Probabilidade de ser solo preto: {black_probability * 100:.2f}%")
    st.write(f"Probabilidade de ser solo vermelho: {red_probability * 100:.2f}%")
    