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


# ---------------------------------------------------------------------------
# c) Teste de Friedman e pós-teste de Nemenyi
# ---------------------------------------------------------------------------

# métricas em que o MENOR valor é o melhor
MENOR_MELHOR = {'erro'}


def friedman_nemenyi(resultados, indice_metrica, alfa=0.05):
    """Friedman + Nemenyi para uma métrica.

    resultados     : dict {nome: res}, res de forma (N, 4), N = número de folds
    indice_metrica : 0 erro, 1 precisão, 2 cobertura, 3 F-measure

    Todos os classificadores precisam ter sido avaliados nos MESMOS folds,
    na mesma ordem: o Friedman é pareado — cada fold é um bloco.
    """
    nomes = list(resultados)
    k = len(nomes)
    metrica = METRICAS[indice_metrica]

    # matriz N x k: uma linha por fold, uma coluna por classificador
    M = np.column_stack([np.asarray(resultados[n])[:, indice_metrica]
                         for n in nomes])
    N = len(M)

    # ranks dentro de cada fold: 1 = melhor (empates recebem o rank médio)
    sinal = 1.0 if metrica in MENOR_MELHOR else -1.0
    ranks = np.vstack([stats.rankdata(sinal * linha) for linha in M])
    rank_medio = ranks.mean(axis=0)

    # Friedman: H0 = os k classificadores são equivalentes
    chi2, p_chi2 = stats.friedmanchisquare(*[M[:, j] for j in range(k)])

    # correção de Iman-Davenport (menos conservadora que o qui-quadrado)
    denom = N * (k - 1) - chi2
    F = (N - 1) * chi2 / denom if denom > 0 else np.inf
    p_F = stats.f.sf(F, k - 1, (k - 1) * (N - 1)) if denom > 0 else 0.0

    # Nemenyi: distância crítica entre ranks médios
    q_alfa = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    CD = q_alfa * np.sqrt(k * (k + 1) / (6.0 * N))

    dif = np.abs(rank_medio[:, None] - rank_medio[None, :])

    return {
        'metrica': metrica, 'nomes': nomes, 'N': N, 'k': k, 'alfa': alfa,
        'media': M.mean(axis=0), 'rank_medio': rank_medio,
        'chi2': chi2, 'p_chi2': p_chi2, 'F': F, 'p_F': p_F,
        'q_alfa': q_alfa, 'CD': CD,
        'diferencas': dif, 'significativo': dif > CD,
    }


def _p(valor):
    # p-valores abaixo do menor float representável saem como 0.0
    return '< 1e-300' if valor == 0 else f'{valor:.3e}'


def imprimir_friedman(s):
    barra = '=' * 70
    print(f'\n{barra}\nFriedman + Nemenyi — {s["metrica"]} '
          f'({s["k"]} classificadores, {s["N"]} folds)\n{barra}')

    print(f'  Friedman:        qui² = {s["chi2"]:9.3f}   p = {_p(s["p_chi2"])}')
    print(f'  Iman-Davenport:  F    = {s["F"]:9.3f}   p = {_p(s["p_F"])}')
    rejeita = s['p_F'] < s['alfa']
    print(f'  -> {"rejeita" if rejeita else "NÃO rejeita"} H0 '
          f'(classificadores equivalentes) com alfa = {s["alfa"]}')

    larg = max(len(n) for n in s['nomes'])
    print(f'\n  {"classificador":<{larg}} | rank médio | {"média":>7}')
    print('  ' + '-' * (larg + 24))
    for j in np.argsort(s['rank_medio']):
        print(f'  {s["nomes"][j]:<{larg}} | {s["rank_medio"][j]:10.3f} | '
              f'{s["media"][j]:7.4f}')

    if not rejeita:
        print('\n  Friedman não rejeitou H0: o pós-teste não se aplica.')
        return

    print(f'\n  Nemenyi: q_alfa = {s["q_alfa"]:.3f}, CD = {s["CD"]:.4f}')
    print('  diferença de rank médio por par (* = significativa, |dif| > CD):')
    ordem = list(np.argsort(s['rank_medio']))
    for a_pos, a in enumerate(ordem):
        for b in ordem[a_pos + 1:]:
            marca = '*' if s['significativo'][a, b] else ' '
            print(f'   {marca} {s["nomes"][a]:<{larg}} x {s["nomes"][b]:<{larg}}'
                  f'  {s["diferencas"][a, b]:.3f}')


def testar_todas(resultados, alfa=0.05, imprimir=True):
    """Friedman + Nemenyi para as quatro métricas. Devolve {métrica: saída}."""
    saidas = {}
    for j, m in enumerate(METRICAS):
        saidas[m] = friedman_nemenyi(resultados, j, alfa)
        if imprimir:
            imprimir_friedman(saidas[m])
    return saidas
