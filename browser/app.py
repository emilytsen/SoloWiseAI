import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os

# Carregar modelo treinado
loaded_model = load_model('/Users/emilytsen/Documents/fatec/SoloWiseAI/TG/soil_classifier_model.h5')

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

# Pasta local onde as imagens estão armazenadas
local_image_folder = "/Users/emilytsen/Documents/fatec/SoloWiseAI/datasets/sand/test"

# Listar as imagens na pasta local
image_files = [f for f in os.listdir(local_image_folder) if os.path.isfile(os.path.join(local_image_folder, f))]

# Permitir ao usuário selecionar uma imagem
selected_image = st.selectbox("Escolha uma imagem...", image_files)

# Adicionar um botão para atualizar a análise
if st.button("Analisar Imagem"):
    if selected_image:
        local_file_path = os.path.join(local_image_folder, selected_image)

        # Prever solo
        black_probability, red_probability = predict_soil_type_with_prob(local_file_path)
        if black_probability > red_probability:
            predicted_class = "Preto"
            higher_probability = black_probability
            plant_suggestion = "Plantas como milho, feijão e soja são bem adaptadas ao solo preto."
        else:
            predicted_class = "Vermelho"
            higher_probability = red_probability
            plant_suggestion = "Plantas como tomate, morango e pimentão são bem adaptadas ao solo vermelho."

        # Exibir resultados na interface
        st.image(local_file_path, caption="Imagem de Solo", width=300)

        # Criar e exibir uma tabela com as informações
        results_table = pd.DataFrame({
            "Classe do solo": [predicted_class],
            "Probabilidade de ser solo preto": [f"{black_probability * 100:.2f}%"],
            "Probabilidade de ser solo vermelho": [f"{red_probability * 100:.2f}%"]
        })
        st.table(results_table)
        st.write(f"\nRecomendação de plantio: {plant_suggestion}")

        if predicted_class == "Vermelho":
            st.title("Características do Solo Vermelho")
            with st.expander("Cor"):
                st.write("Como o nome sugere, o solo vermelho tem uma coloração avermelhada devido à presença de óxidos de ferro, especialmente o óxido de ferro (III) ou ferrugem.")
            with st.expander("Textura"):
                st.write("A textura do solo vermelho pode variar, mas muitas vezes é composta por partículas finas e bem desagregadas. Essa textura pode afetar a capacidade de retenção de água e nutrientes do solo.")
            with st.expander("Composição Mineral"):
                st.write("O solo vermelho geralmente contém minerais de argila, areia e silte. A proporção desses minerais pode variar dependendo da região geográfica.")
            with st.expander("Fertilidade"):
                st.write("O solo vermelho pode ser naturalmente fértil devido à presença de minerais de ferro e outros nutrientes. No entanto, a fertilidade do solo pode variar dependendo de fatores como a quantidade de matéria orgânica e a disponibilidade de água.")
            with st.expander("Oxidação"):
                st.write("O solo vermelho é frequentemente associado à oxidação do ferro, que ocorre quando minerais de ferro são expostos ao oxigênio e à água. Isso pode resultar na formação de manchas ou veios vermelhos no solo.")
            with st.expander("Resistência à Erosão"):
                st.write("Devido à sua textura fina e agregação, o solo vermelho pode ser mais suscetível à erosão, especialmente em áreas com chuvas intensas ou ventos fortes.")
            with st.expander("Distribuição Geográfica"):
                st.write("O solo vermelho pode ser encontrado em várias partes do mundo, incluindo regiões tropicais e subtropicais. Sua distribuição geográfica pode ser influenciada por fatores climáticos, geológicos e pedogenéticos.")
            with st.expander("Plantio Apropriado"):
                st.write("""
                    - **Café**: O solo vermelho é altamente valorizado para o cultivo de café devido à sua boa drenagem e capacidade de reter nutrientes.
                    - **Frutas**: Frutíferas como laranja, uva e abacate se desenvolvem muito bem nesse tipo de solo, especialmente em climas tropicais.
                    - **Soja**: A soja é outra cultura que se beneficia das características do solo vermelho, sendo comum em regiões agrícolas produtivas.
                    - **Milho**: Ideal para o cultivo de milho, que exige um solo bem drenado e com boa capacidade de retenção de água.
                    - **Algodão**: Como no solo preto, o solo vermelho oferece boas condições para o cultivo de algodão, uma vez que mantém um nível adequado de umidade.
                    - **Hortaliças**: Algumas hortaliças, como tomate e pimentão, também prosperam em solos vermelhos bem drenados e ricos em nutrientes.
                """)


        if predicted_class == "Preto":
            st.title("Características do Solo Preto")
            with st.expander("Cor"):
                st.write("O solo preto tem uma cor escura devido ao alto teor de matéria orgânica em decomposição. Essa matéria orgânica ajuda a reter umidade e nutrientes, tornando-o especialmente fértil.")
            with st.expander("Textura"):
                st.write("Em muitas regiões onde o solo orgânico preto é encontrado em abundância, ele tende a ter uma textura argilosa ou silto-argilosa. Isso ocorre porque a decomposição de matéria orgânica ao longo do tempo pode adicionar uma quantidade significativa de partículas finas ao solo, resultando em uma textura mais argilosa.")
            with st.expander("Composição Mineral"):
                st.write("O solo preto é composto principalmente por minerais argilosos, como a montmorilonita e a caulinita, que são responsáveis pela sua textura densa e alta capacidade de retenção de água e nutrientes.")
            with st.expander("Fertilidade"):
                st.write("Devido ao alto teor de matéria orgânica, o solo preto é altamente fértil e adequado para o cultivo de uma variedade de plantas. Ele fornece nutrientes essenciais às plantas e ajuda a melhorar a estrutura do solo.")
            with st.expander("Oxidação"):
                st.write("A oxidação é um processo que pode afetar a cor e a estrutura do solo. No caso do solo preto, a matéria orgânica presente nele pode passar por processos de oxidação ao longo do tempo, resultando em mudanças na sua cor e na disponibilidade de nutrientes.")
            with st.expander("Resistência à Erosão"):
                st.write("O solo preto, devido à sua alta fertilidade e teor de matéria orgânica, tende a ter uma estrutura mais estável e coesa, o que pode conferir certa resistência à erosão. No entanto, em condições de uso inadequado, como agricultura intensiva ou desmatamento, o solo preto pode ficar vulnerável à erosão.")
            with st.expander("Distribuição Geográfica"):
                st.write("O solo preto é comumente encontrado em regiões de clima tropical, especialmente em áreas com vegetação de savana ou floresta tropical úmida. Alguns exemplos de regiões onde o solo preto é predominante incluem o Cerrado brasileiro, partes da África subsaariana, Índia e Austrália. No Brasil, o solo preto é especialmente associado à região do Planalto Central.")
            with st.expander("Plantio Apropriado"):
                st.write("""
                    - **Grãos**: Culturas como milho, soja e trigo se desenvolvem muito bem nesse tipo de solo, garantindo altos rendimentos.
                    - **Cana-de-açúcar**: Ideal para o cultivo de cana, sendo amplamente utilizado em regiões como São Paulo.
                    - **Algodão**: O solo preto oferece os nutrientes necessários para o desenvolvimento do algodão.
                    - **Café**: Comumente utilizado no cultivo de café, especialmente em regiões como Minas Gerais.
                    - **Hortaliças e Legumes**: Batata, cenoura e outros vegetais que precisam de solos bem drenados e ricos em nutrientes.
                """)