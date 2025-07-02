import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import tf_keras

st.set_page_config(page_title="🧬 Classificação de Câncer de Mama", layout="centered")

# ---------------- MENU ----------------
menu = st.sidebar.radio("Navegação", ["🏠 Diagnóstico", "📚 Sobre a Doença", "📞 Contactos"])

if menu == "🏠 Diagnóstico":
    st.title("🔬 Classificador de Câncer de Mama com IA")
    st.image("cancer.png", caption="Ilustração: Câncer de Mama", use_column_width=True)

    st.markdown("""
Este sistema usa redes neurais profundas com MobileNetV2 para classificar imagens de tecidos mamários em três categorias:

- 🟢 **Benigno**
- 🔴 **Maligno**
- ⚪ **Normal**

Faça upload de uma imagem para obter o diagnóstico automático.
    """)
    st.markdown("---")

    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url, trainable=False, input_shape=(224, 224, 3))

    model = tf_keras.Sequential([
        feature_extractor_layer,
        tf_keras.layers.Dense(3, activation='softmax')
    ])
    model.load_weights('breast_classification_model.h5')

    uploaded_file = st.file_uploader("📁 Selecione uma imagem de tecido mamário (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Imagem Carregada", use_column_width=True)

        with st.spinner("🔍 Analisando imagem..."):
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            if image.shape[-1] == 4:
                image = image[..., :3]
            input_tensor = np.expand_dims(image, axis=0)

            prediction = model.predict(input_tensor)
            prediction_label = np.argmax(prediction)
            confidence = np.max(prediction)

            labels = ["🟢 Benigno", "🔴 Maligno", "⚪ Normal"]
            result = labels[prediction_label]

        st.success(f"✅ **Resultado:** {result}")
        st.info(f"📊 Confiança: **{confidence * 100:.2f}%**")

elif menu == "📚 Sobre a Doença":
    st.title("📚 Informações sobre o Câncer de Mama")
    st.image("cancer.png", use_column_width=True)
    st.markdown("""
O **câncer de mama** é uma doença caracterizada pela multiplicação desordenada de células anormais na mama, que formam um tumor maligno.

### Fatores de risco:
- Histórico familiar
- Idade avançada
- Exposição a hormônios
- Obesidade e alcoolismo

### Sintomas comuns:
- Nódulo na mama
- Secreção anormal
- Mudança na pele ou formato do seio

### Prevenção:
- Autoexame regular
- Mamografia periódica
- Estilo de vida saudável

⚠️ **A detecção precoce aumenta significativamente as chances de cura.**
    """)

elif menu == "📞 Contactos":
    st.title("📞 Contactos e Apoio")
    st.markdown("""
Para mais informações ou suporte, entre em contacto:

- 📧 **Email:** apoio@saude.gov.ao
- ☎️ **Linha Saúde:** 111
- 🌐 **Website:** [www.min-saude.gov.ao](http://www.min-saude.gov.ao)

Instituições de apoio:
- Associação Angolana de Luta Contra o Câncer
- Unidades hospitalares com serviço de oncologia

🕊️ Estamos juntos na luta contra o câncer de mama.
    """)
