import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from double_kmeans import Double_KMeans, avaliar_pares
from plots import (plot_silhueta_por_par, plot_sil_vs_ari,
                   plot_silhueta_objetos, resultado_final)

RAIZ = Path(__file__).resolve().parents[2]          # pasta AM-1
DADOS = RAIZ / 'data'
RESULTADOS = Path(__file__).resolve().parent / 'results'
RESULTADOS.mkdir(exist_ok=True)

RODAR = False    # True: roda as 9 x 100 execuções. False: carrega de results/

nomes = []
with open(DADOS / 'spambase.names') as arq:
    for linha in arq:
        linha = linha.strip()
        if linha.endswith('continuous.'):
            nomes.append(linha.split(':')[0].strip())

nomes.append('spam')

spam_data = pd.read_csv(DADOS / 'spambase.data', header=None, names=nomes)

X = spam_data.iloc[:, :-1].values
y = spam_data.iloc[:, -1].values


arq_resultados = RESULTADOS / 'double_kmeans_X.pkl'

if RODAR or not arq_resultados.exists():
    pares = [(K, H) for K in (2, 3, 4) for H in range(1, K + 1)]
    resultados = {}

    t0 = time.time()
    for idx, (K, H) in enumerate(pares):
        dk = Double_KMeans(K=K, H=H, n_init=100, random_state=idx * 1000).fit(X)

        resultados[(K, H)] = {
            "row_labels": dk.row_labels_, "col_labels": dk.col_labels_,
            "G": dk.G_, "W": dk.W_, "hist": dk.hist_, "todos_W": dk.Ws_,
        }

        print(f"K={K} H={H} | W={dk.W_:11.2f} | "
              f"pior={dk.Ws_.max():11.2f} | "
              f"solucoes distintas={len(set(np.round(dk.Ws_, 2)))} | "
              f"{time.time()-t0:5.1f}s")

    with open(arq_resultados, 'wb') as arq:
        pickle.dump(resultados, arq)
else:
    with open(arq_resultados, 'rb') as arq:
        resultados = pickle.load(arq)

melhor_par = avaliar_pares(resultados, X, y, nome_matriz='X')

# rótulos da partição do K*: viram a variável resposta da 2ª versão da Questão 2
rotulos = resultados[melhor_par]['row_labels']
np.save(RESULTADOS / 'rotulos_kestrela.npy', rotulos)
with open(RESULTADOS / 'kestrela.json', 'w') as arq:
    json.dump({'K': int(melhor_par[0]), 'H': int(melhor_par[1]),
               'tamanhos': np.bincount(rotulos).tolist()}, arq)
print(f'\nrótulos do K* = {melhor_par[0]} salvos em {RESULTADOS} '
      f'(tamanhos dos grupos: {np.bincount(rotulos).tolist()})')

plot_silhueta_por_par(resultados, matriz='X')
plot_sil_vs_ari(resultados, matriz='X')
plot_silhueta_objetos(resultados, X, melhor_par, matriz='X')
resultado_final(resultados, y, melhor_par, matriz='X')