"""Questão 2, item d) — curva de aprendizagem.

Conjuntos de treinamento e teste de (5%, 95%) a (95%, 5%), passo de 5%,
com amostragem estratificada. Para cada par, os hiperparâmetros são
reajustados por CV interna no treino, e as métricas são computadas no
conjunto de treinamento E no de teste.
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.k_folds import int_kfold
from utils.metrics import metricas
from models.maj_vote import VotoMajoritario

FRACOES = np.round(np.arange(1, 20) / 20, 2)       # 0.05, 0.10, ..., 0.95
NOME_VOTO = 'voto majoritário'


def _curto(nome):
    # 'bayesiano parzen' -> 'parzen', 'knn bayesiano' -> 'knn'
    return nome.replace('bayesiano', '').strip().split()[0][:6]


def curva_aprendizagem(X, y, classificadores, fracoes=FRACOES, cv_interna=5,
                       media='macro', seed=42, imprimir=True):
    """classificadores: o mesmo dict CLASSIFICADORES do main,
                        {sigla: (nome, grade, criar_modelo)}.

    Devolve um dict com:
      'fracoes'  : (n_p,)          fração usada no treino
      'n_treino' : (n_p,)          número de exemplos de treino
      'treino'   : {nome: (n_p, 4)} erro, precisão, cobertura, F no TREINO
      'teste'    : {nome: (n_p, 4)} idem no TESTE
      'hiper'    : {nome: [dict, ...]} hiperparâmetro escolhido em cada fração
    O voto majoritário entra como quinto classificador.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    nomes = [c[0] for c in classificadores.values()] + [NOME_VOTO]
    treino = {n: [] for n in nomes}
    teste = {n: [] for n in nomes}
    hiper = {c[0]: [] for c in classificadores.values()}
    n_treino = []
    t0 = time.time()

    for p in fracoes:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, train_size=float(p), stratify=y, random_state=seed)

        esc = StandardScaler().fit(Xtr)                # ajusta só no treino
        Xtr, Xte = esc.transform(Xtr), esc.transform(Xte)

        preds_tr, preds_te = [], []
        for nome, grade, criar_modelo in classificadores.values():
            hp = int_kfold(Xtr, ytr, grade, criar_modelo, cv=cv_interna,
                           seed=seed, imprimir=False)
            clf = criar_modelo(hp).fit(Xtr, ytr)
            p_tr, p_te = clf.predict(Xtr), clf.predict(Xte)

            treino[nome].append(metricas(ytr, p_tr, media))
            teste[nome].append(metricas(yte, p_te, media))
            hiper[nome].append(hp)
            preds_tr.append(p_tr)
            preds_te.append(p_te)

        # voto majoritário sobre os quatro classificadores já ajustados
        voto = VotoMajoritario().fit(None, ytr)
        treino[NOME_VOTO].append(metricas(ytr, voto.votar(preds_tr), media))
        teste[NOME_VOTO].append(metricas(yte, voto.votar(preds_te), media))

        n_treino.append(len(ytr))
        if imprimir:
            f_te = '  '.join(f'{_curto(n)}={teste[n][-1][3]:.3f}' for n in nomes)
            print(f'treino {p:4.0%} (n={len(ytr):4d}) | F teste: {f_te} '
                  f'| {time.time() - t0:5.0f}s', flush=True)

    return {
        'fracoes': np.asarray(fracoes, dtype=float),
        'n_treino': np.array(n_treino),
        'treino': {n: np.array(v) for n, v in treino.items()},
        'teste': {n: np.array(v) for n, v in teste.items()},
        'hiper': hiper,
    }