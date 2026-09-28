import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from utils.metrics import metricas

def int_kfold(X, y, grade, criar_modelo, cv=5, seed=0, imprimir=True, rotulos=None):
    """CV estratificada para escolher a melhor configuração da grade.

    grade        : lista de dicionários de configuração
    criar_modelo : função (dict) -> modelo construído, com fit e predict
    rotulos      : nomes curtos das configurações, para a tabela
    """
    acertos = np.zeros((cv, len(grade)))
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)

    for i_fold, (tr, va) in enumerate(skf.split(X, y)):
        if imprimir:
            print(f'Rodando fold {i_fold+1}/{cv} interno')

        for i, g in enumerate(grade):
            try:
                clf = criar_modelo(g).fit(X[tr], y[tr])
                acertos[i_fold, i] = (clf.predict(X[va]) == y[va]).mean()
            except np.linalg.LinAlgError:
                acertos[i_fold, i] = -np.inf      # configuração inválida

    melhor = int(acertos.sum(axis=0).argmax())

    if imprimir:
        rot = rotulos or [str(g) for g in grade]
        larg = max(9, max(len(r) for r in rot) + 1)
        print(' ' * 8 + ''.join(f'{r:>{larg}}' for r in rot))
        for i_f in range(cv):
            print(f'fold {i_f+1:<3}' + ''.join(f'{a:>{larg}.4f}' for a in acertos[i_f]))
        print(' ' * 8 + '-' * (larg * len(grade)))
        print('média   ' + ''.join(f'{a:>{larg}.4f}' for a in acertos.mean(axis=0)))
        print(f'melhor: {grade[melhor]}')

    return grade[melhor]


def ext_kfold(X, y, grade, criar_modelo, nome='', n_rep=30, n_folds=10,
              cv_interna=5, media='macro', imprimir=True):
    """Protocolo 30 x 10-folds estratificado.

    Devolve (res, hiper, preds, ytes):
      res   : (n_rep*n_folds, 4) — erro, precisão, cobertura, F-measure
      hiper : configuração escolhida em cada fold
      preds : predições no fold de teste (para o voto majoritário)
      ytes  : rótulos verdadeiros do fold de teste
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    res, hiper, preds, ytes = [], [], [], []
    t0 = time.time()

    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep)

        for i_fold, (tr_ext, te_ext) in enumerate(skf.split(X, y)):
            esc = StandardScaler().fit(X[tr_ext])
            Xtr, Xte = esc.transform(X[tr_ext]), esc.transform(X[te_ext])
            ytr, yte = y[tr_ext], y[te_ext]

            hp = int_kfold(Xtr, ytr, grade, criar_modelo, cv=cv_interna,
                           seed=rep * 100 + i_fold, imprimir=False)
            clf = criar_modelo(hp).fit(Xtr, ytr)
            pred = clf.predict(Xte)
            m = metricas(yte, pred, media)

            if imprimir:
                print(f'{nome} | rep {rep+1}/{n_rep} | fold {i_fold+1}/{n_folds} '
                      f'| {hp} | erro={m[0]:.4f}')

            res.append(m)
            hiper.append(hp)
            preds.append(pred)
            ytes.append(yte)

        if imprimir:
            print(f'--- {nome}: repetição {rep+1}/{n_rep} ({time.time()-t0:.0f}s) ---')

    return np.array(res), hiper, preds, ytes




