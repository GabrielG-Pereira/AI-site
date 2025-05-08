from flask import Flask, request, jsonify
from AI_models.binary_classification_explainer import load_model, model_predict, explain_binary_classification_model

api = Flask(__name__)

@api.route('/api/rede-neural-simples', methods=['POST'])
def rede_neural_simples():
    model_path = 'AI_models/binary_classification_model.keras'
    data_path = 'data/synthetic_streaming_behavior_dataset.csv'

    response = dict()

    data = request.get_json()

    model = load_model(model_path)
    response['prediction'] = model_predict(model=model, row=data['input'])
    response['explanation'] = explain_binary_classification_model(data_path=data_path, model=model, row=data['input'])

    return jsonify(response), 200

if __name__ == '__main__':
    api.run(debug=True)
