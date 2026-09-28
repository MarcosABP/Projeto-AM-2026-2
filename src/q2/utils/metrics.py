from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from collections import Counter
from scipy import stats


METRICAS = ('erro', 'precisão', 'cobertura', 'F-measure')


def metricas(y_true, y_pred, media='macro'):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=media, zero_division=0
    )
    return np.array([(y_true != y_pred).mean(), p, r, f])

def tabela_comparativa(resultados, conf=0.95, corrigido=False, n_folds=10):
    """resultados: dict {nome: res}, com res de forma (N, 4).
    Imprime média ± semi-amplitude do IC para cada classificador e métrica."""
    larg = max(len(n) for n in resultados)
    titulo = 'IC corrigido (Nadeau-Bengio)' if corrigido else 'IC padrão'
    cab = f'{"classificador":<{larg}} | ' + ' | '.join(f'{m:^15}' for m in METRICAS)
    print(f'\n{"=" * len(cab)}\ncomparação final — média ± {conf:.0%} {titulo}'
          f'\n{"=" * len(cab)}')
    print(cab)
    print('-' * len(cab))

    tabela = {}
    for nome, res in resultados.items():
        res = np.asarray(res)
        N = len(res)
        media = res.mean(axis=0)
        var = res.var(axis=0, ddof=1)
        var = var * (1 + N / (n_folds - 1)) / N if corrigido else var / N
        meio = stats.t.ppf(0.5 + conf / 2, N - 1) * np.sqrt(var)
        tabela[nome] = (media, meio)
        celulas = ' | '.join(f'{media[j]:.4f} ± {meio[j]:.4f}' for j in range(4))
        print(f'{nome:<{larg}} | {celulas}')
    return tabela



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



