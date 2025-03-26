# Código usado pra gerar o dataset de treinamento da rede neural

##################################################################
# Cenário: Detecção de Comportamento em um App de Streaming (1 dia de uso)
# 12 Inputs binários (0 = não, 1 = sim):
#   0. Assistiu mais de 5 episódios seguidos
#   1. Pulou introdução várias vezes
#   2. Pausou os vídeos muitas vezes
#   3. Assistiu de madrugada
#   4. Assistiu em mais de 2 dispositivos diferentes
#   5. Entrou no app várias vezes no dia
#   6. Mais de 2h de inatividade
#   7. Deu like ou favoritou mais de 3 vezes
#   8. Tentativas de login falhas consecutivas
#   9. Acesso de uma rede pública
#   10. Sessões extremamente curtas e repetidas
#   11. Fez login em redes diferentes
# 4 Outputs binários (0 = não, 1 = sim):
#   0. Usuário dormiu assistindo
#   1. Usuário engajado
#   2. Comportamento suspeito
#   3. Compartilhamento de conta
#################################################################

import numpy as np
import pandas as pd
import os

headers = [
    'maisDe5',
    'pulouIntro',
    'pausas',
    'madrugada',
    'dispositivosDiferentes',
    'entrouNoAppVariasVezes',
    'inatividade',
    'favoritou',
    'falhasLogin',
    'redePublica',
    'sessoesCurtas',
    'redesDiferentes',
    'dormiu',
    'engajado',
    'suspeito',
    'compartilhamento'
]


# Definindo quantidade de amostras
n_samples = 1000

# Gerando 12 inputs binários simulando comportamentos
np.random.seed(42)
data = np.random.randint(0, 2, (n_samples, 12))

# Definindo as 4 saídas binárias com base em padrões intuitivos
outputs = []
for row in data:
    sleeping_user = int((row[0] or row[3]) and row[6])
    engaged_user = int( (row[0] and not row[6]) or np.sum( [ row[1], row[2], row[5], row[7] ] ) >= 3  )
    suspicious_behavior = int( np.sum( [ row[4], row[8] * 2, row[9], row[10], row[11] ] ) >= 4 )
    account_sharing = int(row[4] and row[11])
    outputs.append([sleeping_user, account_sharing, suspicious_behavior, account_sharing])

outputs = np.array(outputs)

# Juntando inputs e outputs em um DataFrame
columns = headers
dataset = np.hstack((data, outputs))
df = pd.DataFrame(dataset, columns=columns)

# Salvando o dataset em CSV
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_streaming_behavior_dataset.csv')
df.to_csv(data_path, index=False)

print("Dataset sintético criado e salvo com sucesso!")
