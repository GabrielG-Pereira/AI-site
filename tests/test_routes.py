import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api import api

def load_test_data(file_path):
    path = os.path.join(os.path.dirname(__file__), file_path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_rede_neural_simples():
    client = api.test_client()
    data = load_test_data('test_input.json')

    response = client.post('/api/rede-neural-simples', data=json.dumps(data), content_type='application/json')

    assert response.status_code == 200
    json_response = response.get_json()
    assert 'prediction' in json_response
    assert 'explanation' in json_response