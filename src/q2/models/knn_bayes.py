import numpy as np
from scipy.spatial.distance import cdist


class BayesKNN:
    """Classificador bayesiano baseado em k-vizinhos.

    Estimativas (slides de k-NN):
        p(x|w_j) = (k_j / n_j) / V      P(w_j) = n_j / n
    logo
        p(x|w_j) P(w_j) = k_j / (n V)

    n e V são iguais para todas as classes e cancelam na comparação:
    a regra bayesiana vira o voto majoritário entre os k vizinhos, e a
    posteriori estimada é P(w_j|x) = k_j / k.

    metric: 'euclidean', 'cityblock' (City-Block) ou 'chebyshev'.
    Empates de DISTÂNCIA na fronteira do k-ésimo vizinho: todos os pontos a
    distância r_k entram na vizinhança (a esfera de volume V os contém),
    então a contagem pode passar de k. Empates no VOTO são resolvidos pela
    maior priori.
    """

    def __init__(self, metric, k=5):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        # método preguiçoso: o treino apenas guarda os dados
        self.X_ = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.y_idx_ = np.searchsorted(self.classes_, y)
        self.priori_ = np.bincount(self.y_idx_) / len(y)
        return self

    def predict_proba(self, X):
        D = cdist(np.asarray(X, dtype=float), self.X_, metric=self.metric)

        # r_k: distância do k-ésimo vizinho mais próximo
        r_k = np.partition(D, self.k - 1, axis=1)[:, self.k - 1]

        # vizinhança = todos os pontos dentro da esfera de raio r_k,
        # inclusive os empatados na fronteira (comuns em Chebyshev)
        dentro = D <= r_k[:, None] * (1 + 1e-12)

        um_quente = np.eye(len(self.classes_))[self.y_idx_]   # (n, c)
        k_j = dentro @ um_quente                               # contagem por classe
        return k_j / k_j.sum(axis=1, keepdims=True)            # P(w_j|x) = k_j / k

    def predict(self, X):
        P = self.predict_proba(X)
        # desempate pela maior priori: soma um valor pequeno demais
        # para inverter uma diferença de um voto inteiro
        P = P + 1e-9 * self.priori_
        return self.classes_[P.argmax(axis=1)]