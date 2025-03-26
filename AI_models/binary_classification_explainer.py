from keras.src.saving import load_model
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import os
import json

def explain_binary_classification_model(data_path, row=None): # row: np.array com 12 valores binários
    # Dataset usado para treinar a rede neural
    df = pd.read_csv(data_path)
    X = df.iloc[:,0:12].to_numpy()
    y = df.iloc[:,12:16].to_numpy()

    # Ajuste conforme o seu modelo e dados
    explainer = LimeTabularExplainer(
        training_data=X,  # Dados de treino
        mode='classification',  # Tipo de problema (classificação)
        training_labels=y
    )

    # Carregar o modelo
    model = load_model('binary_classification_model.keras')

    if row is None:
        row = X[1]

    # Explicar uma previsão
    exp = explainer.explain_instance(row, model.predict)
    explanation_dict = exp.as_list()  # Ou exp.as_map(), dependendo da estrutura que você quer
    explanation_json = json.dumps(explanation_dict)
    return explanation_json

if __name__ == '__main__':
    # from timeit import timeit
    # execution_time = timeit(explain_binary_classification_model, number=1)
    # print(f'TEMPO DE EXECUÇÃO: {execution_time}') # levou 12 segundos

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_streaming_behavior_dataset.csv')
    print(explain_binary_classification_model(data_path))