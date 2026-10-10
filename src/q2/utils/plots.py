"""Questão 2 — figuras dos itens a), b) e c).

Todas as funções só salvam em src/q2/imgs/ (não abrem janela).
SUFIXO é definido pelo main (ex.: '_kestrela') e vai no fim do nome do arquivo.
Cada classificador tem cor fixa em todas as figuras (a cor segue o
classificador, nunca a posição dele no ranking).
"""

from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils.metrics import METRICAS

IMGS = Path(__file__).resolve().parent.parent / 'imgs'
SUFIXO = ''        # trocado pelo main conforme a versão da base

# paleta categórica validada (ordem fixa; segura para daltonismo)
CORES_PALETA = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
TINTA = '#0b0b0b'          # texto principal
TINTA_2 = '#52514e'        # texto secundário, eixos
GRADE = '#e4e3df'          # grade recessiva

plt.rcParams.update({
    'axes.edgecolor': TINTA_2, 'axes.labelcolor': TINTA, 'axes.titlecolor': TINTA,
    'xtick.color': TINTA_2, 'ytick.color': TINTA_2, 'text.color': TINTA,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'font.size': 9.5,
})


def _cores(nomes):
    """Cor fixa por classificador, na ordem em que aparecem."""
    return {n: CORES_PALETA[i % len(CORES_PALETA)] for i, n in enumerate(nomes)}


def _salvar(fig, nome):
    IMGS.mkdir(parents=True, exist_ok=True)
    caminho = IMGS / f'{nome}{SUFIXO}.png'
    fig.savefig(caminho, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'salvo em {caminho}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# a) distribuição das métricas nos 300 folds
# ---------------------------------------------------------------------------

def plot_distribuicao_folds(resultados, arquivo='q2_a_distribuicao_folds'):
    """Boxplot dos valores de cada métrica nos N folds, por classificador.

    resultados: dict {nome: res}, res de forma (N, 4).
    """
    nomes = list(resultados)
    cores = _cores(nomes)
    N = len(next(iter(resultados.values())))

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for j, (ax, metrica) in enumerate(zip(axs.flat, METRICAS)):
        dados = [np.asarray(resultados[n])[:, j] for n in nomes]
        pos = np.arange(len(nomes))[::-1]                 # 1º classificador no topo
        bp = ax.boxplot(dados, positions=pos, vert=False, widths=0.55,
                        patch_artist=True, showfliers=True,
                        medianprops=dict(color=TINTA, lw=1.5),
                        whiskerprops=dict(color=TINTA_2, lw=1),
                        capprops=dict(color=TINTA_2, lw=1),
                        flierprops=dict(marker='o', ms=3, mfc='none',
                                        mec=TINTA_2, alpha=0.6))
        for caixa, n in zip(bp['boxes'], nomes):
            caixa.set_facecolor(cores[n])
            caixa.set_alpha(0.85)
            caixa.set_edgecolor('white')
        # média como losango
        for p, d in zip(pos, dados):
            ax.plot(d.mean(), p, 'D', ms=5, mfc='white', mec=TINTA, mew=1, zorder=4)

        ax.set_yticks(pos)
        ax.set_yticklabels(nomes)
        ax.set_title(metrica)
        ax.grid(axis='x', color=GRADE, lw=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(f'Distribuição das métricas nos {N} folds de teste '
                 f'(30 × 10-folds)  —  ◇ = média', fontsize=12)
    fig.tight_layout()
    _salvar(fig, arquivo)


# ---------------------------------------------------------------------------
# a) hiperparâmetros escolhidos pela validação cruzada interna
# ---------------------------------------------------------------------------

def _rotulo_hp(h):
    if 'penalty' in h:
        return 'sem penalidade' if h['penalty'] is None else f"C = {h['C']:g}"
    if 'gamma' in h:
        return f"γ = {h['gamma']:g}"
    if 'h' in h:
        return f"h = {h['h']:g}"
    if 'k' in h:
        return f"k = {h['k']}, {h['metric']}"
    return str(h)


def _ordem_hp(h):
    """Chave de ordenação pela grade: valores crescentes, 'sem penalidade' no fim."""
    if 'penalty' in h:
        return (1, 0) if h['penalty'] is None else (0, h['C'])
    if 'k' in h:
        return (h['metric'], h['k'])
    return tuple(h.values())


def _pct(f, total):
    return f'{f / total:.1%}' if f / total < 0.01 else f'{f / total:.0%}'


def plot_hiperparametros(hipers, arquivo='q2_a_hiperparametros'):
    """Frequência das configurações escolhidas em cada fold externo.

    hipers: dict {nome: lista de dicionários de hiperparâmetros}.
    """
    nomes = list(hipers)
    cores = _cores(nomes)

    fig, axs = plt.subplots(1, len(nomes), figsize=(4.2 * len(nomes), 3.8))
    for ax, n in zip(np.atleast_1d(axs), nomes):
        cont = Counter(_rotulo_hp(h) for h in hipers[n])
        exemplo = {_rotulo_hp(h): h for h in hipers[n]}
        rotulos = sorted(cont, key=lambda r: _ordem_hp(exemplo[r]))
        freq = [cont[r] for r in rotulos]
        total = sum(freq)
        y = np.arange(len(rotulos))[::-1]
        ax.barh(y, freq, color=cores[n], height=0.7, edgecolor='white', lw=2)
        for yi, f in zip(y, freq):
            ax.text(f + total * 0.01, yi, f'{f}  ({_pct(f, total)})',
                    va='center', fontsize=8.5, color=TINTA_2)
        ax.set_yticks(y)
        ax.set_yticklabels(rotulos)
        ax.set_xlim(0, max(freq) * 1.35)
        ax.set_xlabel('folds')
        ax.set_title(n)
        ax.grid(axis='x', color=GRADE, lw=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(f'Hiperparâmetros escolhidos pela CV interna (5-folds) '
                 f'em cada um dos {total} folds externos', fontsize=12)
    fig.tight_layout()
    _salvar(fig, arquivo)


# ---------------------------------------------------------------------------
# b) estimativa pontual e intervalo de confiança
# ---------------------------------------------------------------------------

def _ic(res, conf, corrigido, n_folds):
    N = len(res)
    var = res.var(axis=0, ddof=1)
    var = var * (1 / N + 1 / (n_folds - 1)) if corrigido else var / N
    return stats.t.ppf(0.5 + conf / 2, N - 1) * np.sqrt(var)


def plot_estimativas_ic(resultados, conf=0.95, n_folds=10,
                        arquivo='q2_b_estimativas_ic'):
    """Média de cada métrica com dois intervalos de confiança:
    o padrão (barra grossa) e o corrigido de Nadeau-Bengio (barra fina)."""
    nomes = list(resultados)
    cores = _cores(nomes)

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    pos = np.arange(len(nomes))[::-1]

    for j, (ax, metrica) in enumerate(zip(axs, METRICAS)):
        for p, n in zip(pos, nomes):
            res = np.asarray(resultados[n])
            m = res[:, j].mean()
            ic_pad = _ic(res, conf, False, n_folds)[j]
            ic_cor = _ic(res, conf, True, n_folds)[j]
            ax.plot([m - ic_cor, m + ic_cor], [p, p], color=cores[n], lw=1.5,
                    solid_capstyle='round')
            ax.plot([m - ic_pad, m + ic_pad], [p, p], color=cores[n], lw=9,
                    solid_capstyle='butt')
            ax.plot(m, p, '|', ms=9, mew=1.5, color='white', zorder=4)
            ax.annotate(f'{m:.4f}', (m, p), xytext=(0, 9),
                        textcoords='offset points', ha='center',
                        fontsize=8, color=TINTA_2)
        ax.set_title(metrica + ('  (menor é melhor)' if metrica == 'erro' else ''))
        ax.set_ylim(pos.min() - 0.6, pos.max() + 0.7)
        ax.grid(axis='x', color=GRADE, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)

    axs[0].set_yticks(pos)
    axs[0].set_yticklabels(nomes)

    # legenda das barras: um exemplo neutro
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=TINTA_2, lw=9, label=f'IC {conf:.0%} padrão'),
           Line2D([], [], color=TINTA_2, lw=1.5, label=f'IC {conf:.0%} corrigido (Nadeau-Bengio)'),
           Line2D([], [], marker='|', ls='', ms=9, mew=1.5, color=TINTA_2,
                  label='média nos folds (traço branco sobre a barra)')]
    fig.legend(handles=leg, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    N = len(next(iter(resultados.values())))
    fig.suptitle(f'Estimativa pontual e intervalo de confiança ({N} folds)',
                 fontsize=12)
    fig.tight_layout()
    _salvar(fig, arquivo)


# ---------------------------------------------------------------------------
# c) diagrama de distância crítica (Nemenyi)
# ---------------------------------------------------------------------------

def _grupos_sem_diferenca(r_ord, cd):
    """Intervalos maximais [i, j] de classificadores (já ordenados por rank)
    cuja maior diferença de rank é menor que a CD."""
    k = len(r_ord)
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and r_ord[j + 1] - r_ord[i] < cd:
            j += 1
        if j > i:
            grupos.append((i, j))
    # remove intervalos contidos em outros
    return [g for g in grupos
            if not any(o != g and o[0] <= g[0] and g[1] <= o[1] for o in grupos)]


def _diagrama_cd(ax, s, cores):
    nomes, rank, cd, k = s['nomes'], s['rank_medio'], s['CD'], s['k']
    ordem = np.argsort(rank)
    r_ord = rank[ordem]
    n_ord = [nomes[i] for i in ordem]

    ax.set_xlim(-0.9, k + 1.9)
    ax.axis('off')

    # régua de ranks: 1 (melhor) à esquerda
    ax.plot([1, k], [0, 0], color=TINTA, lw=1.2)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, 0.1], color=TINTA, lw=1.2)
        ax.text(t, 0.17, str(t), ha='center', va='bottom', fontsize=9)

    # régua da distância crítica
    y_cd = 0.75
    ax.plot([1, 1 + cd], [y_cd, y_cd], color=TINTA, lw=2)
    for x in (1, 1 + cd):
        ax.plot([x, x], [y_cd - 0.07, y_cd + 0.07], color=TINTA, lw=2)
    ax.text(1 + cd / 2, y_cd + 0.12, f'CD = {cd:.3f}', ha='center',
            va='bottom', fontsize=9)

    # classificadores: metade melhor rotulada à esquerda, o resto à direita
    passo = 0.42
    metade = (k + 1) // 2
    for pos_i, (r, n) in enumerate(zip(r_ord, n_ord)):
        if pos_i < metade:
            yy = -passo * (pos_i + 1)
            x_txt, ha = 0.75, 'right'
        else:
            yy = -passo * (k - pos_i)
            x_txt, ha = k + 0.25, 'left'
        ax.plot([r, r], [0, yy], color=TINTA_2, lw=1)
        ax.plot([r, x_txt + (0.05 if ha == 'right' else -0.05)], [yy, yy],
                color=TINTA_2, lw=1)
        ax.plot(r, 0, 'o', ms=7, mfc=cores[n], mec='white', mew=1.5, zorder=5)
        ax.text(x_txt, yy, f'{n} ({r:.2f})', ha=ha, va='center', fontsize=9)

    # barras ligando quem NÃO difere significativamente
    for g_i, (i, j) in enumerate(_grupos_sem_diferenca(r_ord, cd)):
        yy = -0.13 - 0.1 * g_i
        ax.plot([r_ord[i] - 0.04, r_ord[j] + 0.04], [yy, yy],
                color=TINTA, lw=3.5, solid_capstyle='round', zorder=6)

    ax.set_ylim(-passo * metade - 0.35, 1.25)
    ax.set_title(f'{s["metrica"]}', fontsize=11, pad=2)


def plot_nemenyi(testes, arquivo='q2_c_nemenyi'):
    """Diagramas de distância crítica (Demšar, 2006) para as quatro métricas.

    testes: saída de testar_todas(), dict {métrica: resultado}.
    """
    s0 = next(iter(testes.values()))
    cores = _cores(s0['nomes'])

    fig, axs = plt.subplots(2, 2, figsize=(13, 6.2))
    for ax, metrica in zip(axs.flat, METRICAS):
        _diagrama_cd(ax, testes[metrica], cores)

    fig.suptitle(f'Pós-teste de Nemenyi (α = {s0["alfa"]}, N = {s0["N"]} folds) — '
                 f'rank médio, 1 = melhor. Barras pretas ligam classificadores '
                 f'sem diferença significativa.', fontsize=11)
    fig.tight_layout()
    _salvar(fig, arquivo)


# ---------------------------------------------------------------------------
# c) teste de Friedman: distribuição das posições em cada fold
# ---------------------------------------------------------------------------

# rampa sequencial (azul), do claro ao escuro
RAMPA_AZUL = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
              '#256abf', '#184f95', '#0d366b']


def _frequencia_ranks(M, menor_melhor):
    """M: matriz N x k (folds x classificadores) de uma métrica.
    Devolve F (k x k): F[i, r] = fração dos folds em que o classificador i
    ficou na posição r+1. Empates dividem o peso entre as posições que ocupam."""
    N, k = M.shape
    sinal = 1.0 if menor_melhor else -1.0
    F = np.zeros((k, k))
    for linha in sinal * M:
        r_min = stats.rankdata(linha, method='min').astype(int)
        r_max = stats.rankdata(linha, method='max').astype(int)
        for i in range(k):
            t = r_max[i] - r_min[i] + 1
            F[i, r_min[i] - 1:r_max[i]] += 1.0 / t
    return F / N


def plot_friedman(resultados, testes, arquivo='q2_c_friedman'):
    """Mapa de calor: em quantos % dos folds cada classificador ficou em cada
    posição. Sob H0 (classificadores equivalentes), toda célula valeria 1/k.

    resultados: dict {nome: res}; testes: saída de testar_todas().
    """
    from matplotlib.colors import LinearSegmentedColormap
    from utils.metrics import MENOR_MELHOR

    nomes = list(resultados)
    k = len(nomes)
    cmap = LinearSegmentedColormap.from_list('azul', RAMPA_AZUL)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.4))
    for j, (ax, metrica) in enumerate(zip(axs.flat, METRICAS)):
        M = np.column_stack([np.asarray(resultados[n])[:, j] for n in nomes])
        F = _frequencia_ranks(M, metrica in MENOR_MELHOR)

        ax.imshow(F, cmap=cmap, vmin=0, vmax=1, aspect='auto',
                  interpolation='nearest')
        for i in range(k):
            for r in range(k):
                v = F[i, r]
                ax.text(r, i, f'{v:.0%}' if v >= 0.005 else '·',
                        ha='center', va='center', fontsize=9,
                        color='white' if v > 0.45 else TINTA)
        # separadores brancos só entre as células (sem contorno externo)
        for b in np.arange(0.5, k - 0.5):
            ax.axvline(b, color='white', lw=2)
            ax.axhline(b, color='white', lw=2)
        for lado in ax.spines.values():
            lado.set_visible(False)

        ax.set_xticks(range(k))
        ax.set_xticklabels([f'{r + 1}º' for r in range(k)])
        ax.set_yticks(range(k))
        ax.set_yticklabels(nomes)
        ax.tick_params(length=0)
        ax.set_xlabel('posição no fold (1º = melhor)')

        s = testes[metrica]
        p = '< 1e-300' if s['p_chi2'] == 0 else f'{s["p_chi2"]:.1e}'
        ax.set_title(f'{metrica}   —   χ² = {s["chi2"]:.1f},  p = {p}', fontsize=10.5)

    fig.suptitle(f'Teste de Friedman: posição de cada classificador nos {len(M)} folds. '
                 f'Sob H0 (equivalentes), toda célula valeria {1 / k:.0%}.',
                 fontsize=11.5)
    fig.tight_layout()
    _salvar(fig, arquivo)


# ---------------------------------------------------------------------------
# d) curva de aprendizagem
# ---------------------------------------------------------------------------

def plot_curva_aprendizagem(curva, indice=3, arquivo='q2_d_curva_aprendizagem'):
    """Um painel por classificador: métrica no treino (tracejado) e no
    teste (contínuo), em função do % do conjunto usado no treinamento.

    curva : saída de curva_aprendizagem()
    indice: 0 erro, 1 precisão, 2 cobertura, 3 F-measure (o item d pede o F)
    """
    nomes = list(curva['teste'])
    cores = _cores(nomes)
    x = curva['fracoes'] * 100
    metrica = METRICAS[indice]

    fig, axs = plt.subplots(1, len(nomes), figsize=(3.3 * len(nomes), 3.9),
                            sharey=True)
    for ax, n in zip(axs, nomes):
        tr = curva['treino'][n][:, indice]
        te = curva['teste'][n][:, indice]
        ax.fill_between(x, te, tr, color=cores[n], alpha=0.12, lw=0)
        ax.plot(x, tr, '--', color=cores[n], lw=2)
        ax.plot(x, te, '-', color=cores[n], lw=2, marker='o', ms=3.5)
        # rótulos diretos no fim das linhas
        ax.annotate('treino', (x[-1], tr[-1]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8.5,
                    color=TINTA_2)
        ax.annotate('teste', (x[-1], te[-1]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8.5,
                    color=TINTA_2)
        ax.set_title(n)
        ax.set_xlabel('% do conjunto no treino')
        ax.set_xticks([5, 25, 50, 75, 95])
        ax.set_xlim(0, 112)
        ax.grid(color=GRADE, lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_ylabel(metrica)

    fig.suptitle(f'Curva de aprendizagem — {metrica} no treino (tracejado) e no '
                 f'teste (contínuo); a faixa sombreada é a diferença entre eles',
                 fontsize=11.5)
    fig.tight_layout()
    _salvar(fig, f'{arquivo}_{metrica}')


def plot_curva_teste_comparada(curva, indice=3,
                               arquivo='q2_d_curva_teste_comparada'):
    """As curvas de TESTE dos cinco classificadores no mesmo eixo."""
    nomes = list(curva['teste'])
    cores = _cores(nomes)
    x = curva['fracoes'] * 100
    metrica = METRICAS[indice]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for n in nomes:
        ax.plot(x, curva['teste'][n][:, indice], '-', color=cores[n], lw=2,
                marker='o', ms=3.5, label=n)
    ax.set_xlabel('% do conjunto usado no treinamento '
                  '(o restante é o conjunto de teste)')
    ax.set_ylabel(f'{metrica} no teste')
    ax.set_xticks(np.arange(5, 100, 10))
    ax.grid(color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=len(nomes), fontsize=9, handlelength=1.5, columnspacing=1.2)
    ax.set_title(f'Curva de aprendizagem — {metrica} no teste, por classificador')
    fig.tight_layout()
    _salvar(fig, f'{arquivo}_{metrica}')