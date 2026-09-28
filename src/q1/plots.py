import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_samples, confusion_matrix, adjusted_rand_score

IMGS = Path(__file__).resolve().parent.parent / 'q1/imgs'


def _finalizar(fig, nome):
    IMGS.mkdir(parents=True, exist_ok=True)
    caminho = IMGS / f'{nome}.png'
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    print(f'salvo em {caminho}')
    plt.show()


def plot_silhueta_por_par(resultados, matriz='Xz', limiar_alerta=50):
    """Silhueta x (K,H) e tamanho do menor cluster. Requer 'sil' e 'tamanhos'."""
    pares_ord = list(resultados.keys())
    rotulos = [f"({K},{H})" for K, H in pares_ord]
    sils = [resultados[p]["sil"] for p in pares_ord]
    menores = [resultados[p]["tamanhos"].min() for p in pares_ord]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].plot(rotulos, sils, "o-", ms=7)
    i_max = int(np.argmax(sils))
    ax[0].plot(rotulos[i_max], sils[i_max], "o", ms=13, mfc="none",
               mec="red", mew=2, label=f"K* = {pares_ord[i_max][0]}")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_xlabel("(K, H)"); ax[0].set_ylabel("Silhueta")
    ax[0].set_title(f"Silhueta x (K, H) — {matriz}")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].bar(rotulos, menores, color="steelblue")
    ax[1].axhline(limiar_alerta, color="r", ls="--", lw=1)
    ax[1].text(0, limiar_alerta * 1.1, "limiar de alerta", color="r", fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("(K, H)"); ax[1].set_ylabel("menor cluster (log)")
    ax[1].set_title("Tamanho do menor cluster de objetos")
    ax[1].grid(alpha=0.3, axis="y")

    _finalizar(fig, f'q1_silhueta_por_par_{matriz}')


def plot_silhueta_objetos(resultados, X, par, matriz='Xz'):
    """Silhueta por objeto de um par específico. par = (K, H)."""
    K, H = par
    rl = resultados[par]["row_labels"]
    ss = silhouette_samples(X, rl)
    espaco = max(10, int(0.02 * len(ss)))

    fig = plt.figure(figsize=(7, 4.5))
    y0 = 0
    for k in range(K):
        v = np.sort(ss[rl == k])
        plt.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=0.7,
                          label=f"cluster {k} (n={len(v)}, méd={v.mean():.3f})")
        y0 += len(v) + espaco

    plt.axvline(ss.mean(), color="r", ls="--", label=f"média {ss.mean():.3f}")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("s(i)"); plt.yticks([])
    plt.title(f"Silhueta por objeto — par ({K},{H}) — {matriz}")
    plt.legend(fontsize=8)

    _finalizar(fig, f'q1_silhueta_objetos_{K}x{H}_{matriz}')


def plot_sil_vs_ari(resultados, matriz='Xz'):
    """Silhueta e ARI lado a lado. Requer 'sil' e 'ari'."""
    pares_ord = list(resultados.keys())
    rot = [f"({K},{H})" for K, H in pares_ord]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rot, [resultados[p]["sil"] for p in pares_ord], "o-", label="Silhueta")
    ax.plot(rot, [resultados[p]["ari"] for p in pares_ord], "s-",
            label="ARI vs. classe real")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("(K, H)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Silhueta e ARI por par (K, H) — {matriz}")

    _finalizar(fig, f'q1_sil_vs_ari_{matriz}')


def resultado_final(resultados, y, par, matriz='Xz'):
    """Os três itens pedidos: matriz G, matriz de confusão e W x iterações."""
    K, H = par
    r = resultados[par]
    rl, cl, G, W, hist = (r["row_labels"], r["col_labels"],
                          r["G"], r["W"], r["hist"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    lim = np.abs(G).max()
    im = ax[0].imshow(G, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for k in range(G.shape[0]):
        for h in range(G.shape[1]):
            ax[0].text(h, k, f"{G[k,h]:.3f}", ha="center", va="center")
    ax[0].set_title("i) Matriz de protótipos G")
    ax[0].set_xlabel("grupo de variáveis"); ax[0].set_ylabel("cluster de objetos")
    ax[0].set_xticks(range(G.shape[1])); ax[0].set_yticks(range(G.shape[0]))
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    cm = confusion_matrix(y, rl)
    ax[1].imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[1].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1].set_title("ii) Matriz de confusão")
    ax[1].set_xlabel("cluster do algoritmo"); ax[1].set_ylabel("classe a priori")
    ax[1].set_xticks(range(cm.shape[1]))
    if cm.shape[0] == 2:
        ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["não-spam", "spam"])
    else:
        ax[1].set_yticks(range(cm.shape[0]))

    ax[2].plot(range(1, len(hist) + 1), hist, "o-", ms=4)
    ax[2].set_title("iv) Função objetivo W x iterações")
    ax[2].set_xlabel("iteração"); ax[2].set_ylabel("W"); ax[2].grid(alpha=0.3)

    _finalizar(fig, f'q1_resultado_final_{K}x{H}_{matriz}')

    print(f"G =\n{np.round(G, 3)}")
    print(f"\nconfusão:\n{cm}")
    print(f"\nARI = {adjusted_rand_score(y, rl):.4f}")
    print(f"W final = {W:.2f} em {len(hist)} iterações")
    print(f"objetos por cluster: {np.bincount(rl)}")
    print(f"variáveis por grupo: {np.bincount(cl)}")