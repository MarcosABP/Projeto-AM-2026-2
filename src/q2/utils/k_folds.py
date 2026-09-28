import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import Counter
from scipy import stats
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

    grade        : lista de dicionários de configuração
    criar_modelo : função (dict) -> modelo construído, com fit e predict
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    res, hiper = [], []
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
            m = metricas(yte, clf.predict(Xte), media)

            if imprimir:
                print(f'{nome} | rep {rep+1}/{n_rep} | fold {i_fold+1}/{n_folds} '
                      f'| {hp} | erro={m[0]:.4f}')

            res.append(m)
            hiper.append(hp)

    return np.array(res), hiper


METRICAS = ('erro', 'precisão', 'cobertura', 'F-measure')


def resumir(res, hiper=None, nome='', conf=0.95, corrigido=False, n_folds=10):
    """Estimativa pontual, intervalo de confiança e hiperparâmetros escolhidos.

    corrigido : usa a variância de Nadeau e Bengio, que corrige a
                sobreposição entre os treinos dos folds
    """
    res = np.asarray(res)
    N = len(res)
    barra = '=' * 52

    if hiper is not None:
        cont = Counter(str(h) for h in hiper)
        print(f'\n{barra}\nhiperparâmetros escolhidos em {N} folds'
              f'{" — " + nome if nome else ""}\n{barra}')
        for cfg, n in cont.most_common():
            print(f'  {n:3d}x ({n/N:5.1%})  {cfg}')
        print(f'\nmoda: {cont.most_common(1)[0][0]}')

    media = res.mean(axis=0)
    var = res.var(axis=0, ddof=1)
    var = var * (1 + N / (n_folds - 1)) / N if corrigido else var / N
    meio = stats.t.ppf(0.5 + conf / 2, N - 1) * np.sqrt(var)

    print(f'\n{barra}\ndesempenho em {N} folds'
          f'{" — " + nome if nome else ""}\n{barra}')
    for j, m in enumerate(METRICAS):
        print(f'  {m:>10}: {media[j]:.4f} ± {meio[j]:.4f}')

    return media, meio



