import time
from pathlib import Path

import numpy as np
import pandas as pd

from double_kmeans import Double_KMeans, avaliar_pares
from plots import (plot_silhueta_por_par, plot_sil_vs_ari,
                   plot_silhueta_objetos, resultado_final)

DADOS = Path('C:/AM-1/data')         # barra normal funciona no Windows

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

melhor_par = avaliar_pares(resultados, X, y)

plot_silhueta_por_par(resultados)
plot_sil_vs_ari(resultados)
plot_silhueta_objetos(resultados, X, melhor_par)
resultado_final(resultados, y, melhor_par)