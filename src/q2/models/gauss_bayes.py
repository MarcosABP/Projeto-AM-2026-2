import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

GAMMAS = (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5)

class BayesGaussianoMV:

    def __init__(self, gamma=0.0):
        self.gamma = gamma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        self.classes_ = np.unique(y)
        self.d_ = d

        c = len(self.classes_)
        self.log_priori_ = np.empty(c)
        self.mu_ = np.empty((c, d))
        self.chol_ = np.empty((c, d, d))
        self.logdet_ = np.empty(c)

        for i, classe in enumerate(self.classes_):
            Xc = X[y == classe]
            n_i = len(Xc)

            # a) priori de MV: P(w_i) = n_i / n
            self.log_priori_[i] = np.log(n_i / n)

            # b) média e covariância de MV (divisor n_i)
            mu = Xc.mean(axis=0)
            Xd = Xc - mu
            sigma = (Xd.T @ Xd) / n_i

            # shrinkage opcional (gamma = 0 é MV pura)
            if self.gamma > 0:
                sigma = (1 - self.gamma) * sigma + self.gamma * (np.trace(sigma) / d) * np.eye(d)

            L = np.linalg.cholesky(sigma)          # sigma = L @ L.T

            self.mu_[i] = mu
            self.chol_[i] = L
            self.logdet_[i] = 2 * np.log(np.diag(L)).sum()   # log|sigma|

        return self

    def log_verossimilhanca(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(self.classes_)))

        for i in range(len(self.classes_)):
            Z = np.linalg.solve(self.chol_[i], (X - self.mu_[i]).T)
            maha = (Z ** 2).sum(axis=0)            # (x-mu)^T sigma^-1 (x-mu)

            out[:, i] = (-0.5 * self.d_ * np.log(2 * np.pi)
                         - 0.5 * self.logdet_[i]
                         - 0.5 * maha)
        return out

    def predict_proba(self, X):
        log_num = self.log_verossimilhanca(X) + self.log_priori_
        return np.exp(log_num - logsumexp(log_num, axis=1, keepdims=True))

    def predict(self, X):
        log_num = self.log_verossimilhanca(X) + self.log_priori_
        return self.classes_[log_num.argmax(axis=1)]




def ajustar_gaussiano(X, y, cv=5, seed=0, gammas=GAMMAS, imprimir=True):
    """CV estratificada 5-folds sobre os 9 folds de treino, para fixar gamma."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    acertos = np.zeros((cv, len(gammas)))
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)

    for i_fold, (tr, va) in enumerate(skf.split(X, y)):
      print(f'Rodando fold {i_fold+1}/{cv}  interno')
      for i, g in enumerate(gammas):
          try:
              clf = BayesGaussianoMV(gamma=g).fit(X[tr], y[tr])
              acertos[i_fold, i] = (clf.predict(X[va]) == y[va]).mean()
          except np.linalg.LinAlgError:
              acertos[i_fold, i] = -np.inf     # Sigma singular: nunca vence

    melhor = int(acertos.sum(axis=0).argmax())

    if imprimir:
        print(' ' * 8 + ''.join(f'{f"g={g:g}":>10}' for g in gammas))
        for i_f in range(cv):
            print(f'fold {i_f+1:<3}' + ''.join(f'{a:>10.4f}' for a in acertos[i_f]))
        print(' ' * 8 + '-' * (10 * len(gammas)))
        print('média   ' + ''.join(f'{a:>10.4f}' for a in acertos.mean(axis=0)))
        print(f'melhor: gamma = {gammas[melhor]}')

    return {'gamma': gammas[melhor]}