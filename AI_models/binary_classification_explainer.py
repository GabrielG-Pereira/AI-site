from keras.src.saving import load_model
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np
import os
import json

# função que retorna a resposta do modelo pra um conjunto de dados
def model_predict(model, row): # model: get_model  row: lista com 12 valores binários
    row = np.array(row)  # Garante que row seja um NumPy array
    row = row.reshape(1, -1)  # Ajusta o formato para (1, 12)
    prediction = model.predict(row)
    return prediction[0].tolist()

# A função retorna uma explicação detalhada sobre a previsão do modelo para uma instância específica, no formato JSON.
# O JSON contém explicações para cada uma das 4 classes possíveis (label 0 a 3). Para cada classe, há uma lista de tuplas,
# onde cada tupla tem o formato ("X=valor", peso). Aqui, "X" representa o índice da entrada (0 a 11) e "valor" pode ser 0 ou 1.
# O "peso" indica a contribuição dessa característica para a previsão da classe: valores positivos aumentam a probabilidade da classe,
# enquanto valores negativos a reduzem. Exemplo de saída:
# {
#     "0": [["0=1", 0.15], ["2=1", -0.05], ...],
#     "1": [["3=1", 0.20], ["10=1", -0.08], ...],
#     "2": [["5=0", -0.10], ["8=0", 0.07], ...],
#     "3": [["1=0", 0.25], ["6=1", -0.03], ...]
# }
# Esse JSON pode ser usado pelo front-end para exibir explicações sobre como o modelo chegou à decisão para cada classe.
def explain_binary_classification_model(data_path, model, row): # model: get_model  row: lista com 12 valores binários
    # Dataset usado para treinar a rede neural
    df = pd.read_csv(data_path)
    X = df.iloc[:,0:12].to_numpy()
    y = df.iloc[:,12:16].to_numpy()

    # Ajuste conforme o seu modelo e dados
    explainer = LimeTabularExplainer(
        training_data=X,  # Dados de treino
        mode='classification',  # Tipo de problema (classificação)
        training_labels=y,
        categorical_features=list(range(12))
    )

    row = np.array(row)  # Garante que row seja um NumPy array
    row = row.reshape(1, -1)  # Ajusta o formato para (1, 12)

    # Explicar uma previsão
    exp = explainer.explain_instance(row[0], model.predict, labels=list(range(4)))
    explanation_dict = dict()
    for label in range(4):
        explanation_dict[label] = exp.as_list(label=label)  # Ou exp.as_map(), dependendo da estrutura que você quer
    explanation_json = json.dumps(explanation_dict)
    return explanation_json

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_streaming_behavior_dataset.csv')
    model_path = 'binary_classification_model.keras'
    model = load_model(model_path)
    row = [0,0,1,0,1,1,1,0,1,0,1,1]
    print('----------------------------------------')
    print('PREDICTION:   ', model_predict(model, row))
    print(explain_binary_classification_model(data_path, model, row))