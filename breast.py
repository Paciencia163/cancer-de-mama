import pathlib
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Caminho para o diretório de dados
data_dir = pathlib.Path('Dataset')

# Dicionários de arquivos e rótulos
data_dict = {
    'benign': list(data_dir.glob('benign/*')),
    'malignant': list(data_dir.glob('malignant/*')),
    'normal': list(data_dir.glob('normal/*'))
}

data_label = {
    'benign': 0,
    'malignant': 1,
    'normal': 2,
}

# Pré-processamento das imagens
X = []
y = []
for label, images in data_dict.items():
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            X.append(img)
            y.append(data_label[label])

X = np.array(X)
y = np.array(y)

# Dividir em treino/teste e normalizar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Criar o modelo com MobileNetV2
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(224, 224, 3)),
    feature_extractor_layer,
    keras.layers.Dense(3, activation='softmax')
])

# Compilar e treinar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Salvar o modelo treinado
model.save("breast_classification_model.h5")
print("Modelo salvo como 'breast_classification_model.h5'")
