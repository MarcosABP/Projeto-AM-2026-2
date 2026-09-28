import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


class BayesParzen:
    """Classificador bayesiano com janela de Parzen, kernel multivariado
    produto e kernel gaussiano univariado.

    Para cada classe w_j, com n_j exemplos de treino x_t:

        p(x|w_j) = 1/(n_j h^d) SUM_t PROD_{l=1..d} K((x_l - x_tl) / h)
        K(u)     = (2 pi)^(-1/2) exp(-u^2 / 2)          (gaussiano univariado)
        P(w_j)   = n_j / n

    Com K gaussiano, o produto sobre as d coordenadas colapsa:

        PROD_l K(u_l) = (2 pi)^(-d/2) exp(-||x - x_t||^2 / (2 h^2))

    A soma sobre os exemplos é feita em escala logarítmica (logsumexp):
    com d = 57 os termos exp(...) viram zero em ponto flutuante.

    A mesma janela h vale para todos os atributos, o que exige dados
    padronizados (feito dentro de cada fold).
    """

    def __init__(self, h=1.0):
        self.h = h

    def fit(self, X, y):
        # método preguiçoso: guarda os exemplos de cada classe
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.d_ = X.shape[1]
        self.X_classe_ = [X[y == c] for c in self.classes_]
        self.log_priori_ = np.log([len(Xc) / len(X) for Xc in self.X_classe_])
        return self

    def log_verossimilhanca(self, X, bloco=1000):
        """log p(x|w_j) para cada exemplo e classe, matriz (n, c)."""
        X = np.asarray(X, dtype=float)
        h, d = self.h, self.d_
        # log[ (2 pi)^(-d/2) h^(-d) ]
        cte = -0.5 * d * np.log(2 * np.pi) - d * np.log(h)

        out = np.empty((len(X), len(self.classes_)))
        for ini in range(0, len(X), bloco):           # blocos: limita a memória
            Xb = X[ini:ini + bloco]
            for j, Xc in enumerate(self.X_classe_):
                D2 = cdist(Xb, Xc, metric='sqeuclidean')           # ||x - x_t||^2
                out[ini:ini + bloco, j] = (
                    logsumexp(-D2 / (2 * h ** 2), axis=1)           # log SUM_t exp(.)
                    - np.log(len(Xc))                              # 1 / n_j
                    + cte
                )
        return out

    def predict_proba(self, X):
        log_num = self.log_verossimilhanca(X) + self.log_priori_  # log p(x|w)P(w)
        return np.exp(log_num - logsumexp(log_num, axis=1, keepdims=True))

    def predict(self, X):
        log_num = self.log_verossimilhanca(X) + self.log_priori_
        return self.classes_[log_num.argmax(axis=1)]