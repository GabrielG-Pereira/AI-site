# Código usado para criar a rede neural

import numpy as np
import pandas as pd
from keras.src import Sequential
from keras.src.layers import Dense
from keras.src.utils import plot_model
import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_streaming_behavior_dataset.csv')
df = pd.read_csv(data_path)
inputs = df.iloc[:,0:12]
outputs = df.iloc[:,12:16]

# Gerando dados de exemplo (12 inputs binários e 4 outputs binários)
X = inputs
y = outputs

# Criando o modelo
model = Sequential([
    Dense(16, activation='relu', input_shape=(12,)),
    Dense(8, activation='relu'),
    Dense(4, activation='sigmoid')  # Sigmoid para saídas binárias
])

# plot_model(model, to_file='rede_neural_classificacao_binaria.png', show_shapes=True, show_layer_names=True)

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
history = model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Salvando o modelo treinado
model.save('binary_classification_model.keras')

print("Modelo treinado e salvo com sucesso!")
