import numpy as np

from utils.metrics import metricas


class VotoMajoritario:
    """Combinação de classificadores pela regra do voto majoritário.

    Cada classificador dá um voto para a classe que prevê; vence a classe
    mais votada. Empates são resolvidos pela maior priori (slides de
    combinação de classificadores).

    Dois usos:
      1) com modelos:  VotoMajoritario([m1, m2, m3, m4]).fit(X, y).predict(X)
      2) sem modelos:  VotoMajoritario().fit(None, y).votar([pred1, ..., pred4])
         combina predições já calculadas, sem retreinar nada
    """

    def __init__(self, modelos=None):
        self.modelos = modelos

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        self.priori_ = np.bincount(idx, minlength=len(self.classes_)) / len(y)
        if self.modelos is not None:
            for m in self.modelos:
                m.fit(X, y)
        return self

    def contar_votos(self, preds):
        """preds: lista de vetores de rótulos, um por classificador.
        Devolve a matriz (n_exemplos, n_classes) com o número de votos."""
        n = len(preds[0])
        votos = np.zeros((n, len(self.classes_)))
        for p in preds:
            idx = np.searchsorted(self.classes_, np.asarray(p))
            votos[np.arange(n), idx] += 1
        return votos

    def votar(self, preds):
        votos = self.contar_votos(preds)
        # desempate pela maior priori: valor pequeno demais para inverter
        # uma diferença de um voto inteiro
        votos = votos + 1e-9 * self.priori_
        return self.classes_[votos.argmax(axis=1)]

    def predict(self, X):
        if self.modelos is None:
            raise ValueError('sem modelos: use votar(preds)')
        return self.votar([m.predict(X) for m in self.modelos])


def voto_dos_folds(extras, y, media='macro'):
    """Monta o voto majoritário a partir das predições salvas pelo ext_kfold.

    extras: dict {sigla: {'preds': [...], 'ytes': [...]}}, um por classificador,
            todos avaliados nos MESMOS folds.
    y     : vetor de classes completo (para a priori).

    Devolve (res_voto, n_empates): res_voto tem forma (n_folds, 4), como os
    demais; n_empates conta, por fold, os exemplos decididos pela priori.
    """
    siglas = list(extras)
    n_folds = len(extras[siglas[0]]['ytes'])

    # os folds precisam ser os mesmos para todos os classificadores
    for s in siglas[1:]:
        assert len(extras[s]['ytes']) == n_folds, f'{s}: número de folds diferente'
        for f in range(n_folds):
            assert np.array_equal(extras[s]['ytes'][f], extras[siglas[0]]['ytes'][f]), \
                f'{s}: fold {f} diferente — os classificadores não viram os mesmos folds'

    # priori da base inteira: com amostragem estratificada, a do treino de
    # cada fold coincide com ela até a terceira casa decimal
    voto = VotoMajoritario().fit(None, y)

    res, n_empates = [], []
    for f in range(n_folds):
        preds = [extras[s]['preds'][f] for s in siglas]
        yte = extras[siglas[0]]['ytes'][f]

        votos = voto.contar_votos(preds)
        top = np.sort(votos, axis=1)[:, -2:]
        n_empates.append(int((top[:, 0] == top[:, 1]).sum()))

        res.append(metricas(yte, voto.votar(preds), media))

    return np.array(res), np.array(n_empates)