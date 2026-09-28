import numpy as np
from sklearn.metrics import silhouette_samples, adjusted_rand_score, confusion_matrix, silhouette_score


class Double_KMeans:

    def __init__(self, K, H, max_iter=100, n_init=100, random_state=None):
        self.K = K
        self.H = H
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    @staticmethod
    def _resgatar(labels, custo, n_grupos):
        """Garante que nenhum grupo fique vazio, sem que dois grupos
        disputem o mesmo elemento."""
        usados = set()
        for g in range(n_grupos):
            if (labels == g).any():
                continue
            tam = np.bincount(labels, minlength=n_grupos)
            ok = np.array([i not in usados and tam[labels[i]] > 1
                        for i in range(len(labels))])
            if not ok.any():
                continue
            j = int(np.where(ok, custo[:, g], np.inf).argmin())
            labels[j] = g
            usados.add(j)
        return labels

    
    @staticmethod
    def _inicializacao(N, P, K, H, rng):
        rng = np.random.default_rng(rng)
        rl = Double_KMeans._resgatar(rng.integers(0, K, N), np.zeros((N, K)), K)
        cl = Double_KMeans._resgatar(rng.integers(0, H, P), np.zeros((P, H)), H)
        return rl, cl


    @staticmethod
    def _prototipos(X, row_labels, col_labels, K, H):
        G = np.zeros((K, H))

        for k in range(K):
            u_ik = row_labels == k # compara valores do label aleatorio e do real, retorna T ou F. Se objeto está no cluster k, retorna true

            for h in range(H):
                v_ih = col_labels == h
                G[k,h] = X[np.ix_(u_ik, v_ih)].mean() # média valores dos objetos (linhas) e variaveis (colunas) agrupado por cluster
            # np.ix_ devolve apenas índices verdadeiros. Nesse caso, para entrar no cluster, deve pertencer ao cluster de objetos e de variaveis
        return G

    @staticmethod
    def _reat_objetos(G, X, col_labels, K):
        # essa função vai reatribuir os objetos aos seu cluster mais próximo (menor custo)

        N = X.shape[0] # n de objetos
        custo = np.zeros((N, K)) # matriz de custo

        for k in range(K): # para cada cluster, vai calcular custo de cada objeto
            G_k = G[k][col_labels] # mapeando labels de X para valores de G
            #print(G_k) #Ex pra K = 2: G = [[2.0454084  11.41589822] [1.74554559 11.03720019]] e col_labels [1 1 1 .... 0 0]. para cluster 0,  G[0] =  [2.0454084  11.41589822], G_kh = [11.416, 11.416, 11.416, ..., 2.045, 2.045]
            # G_p é o vetor com valores de referencia para um cluster específico
            #print(G_k.shape)

            custo[:, k] = ((X - G_k) ** 2).sum(axis=1) # (xij - gkh) ^ 2 , para cada objeto, tem o custo de ser = 0 e = 1 (k = 0 calcula custo para cluster 0, k= 1 custo cluster 1)
            #print(custo)

        return Double_KMeans._resgatar(custo.argmin(axis=1), custo, K) # lista de novos clusters de cada objeto, dado o menor custo
    
    @staticmethod
    def _reat_variaveis(G, X, row_labels, H): # essa função vai reatribuir as variaveis aos seu cluster mais próximo (menor custo)
        P = X.shape[1] # n de variaveis
        custo = np.zeros((P, H)) # matriz de custo
        for h in range(H):
            G_h = G[:, h][row_labels]
            custo[:, h] = ((X - G_h[:, None]) ** 2).sum(axis=0)
        return Double_KMeans._resgatar(custo.argmin(axis=1), custo, H)

    @staticmethod
    def _parada(X, G, row_labels, col_labels):
        return ((X - G[row_labels][:, col_labels]) ** 2).sum() #criterio de parada

    def _double_k_means(self, X, rng):
        N, P = X.shape
        rng = np.random.default_rng(rng)
        historico_custo = []

        row_labels, col_labels = self._inicializacao(N, P, self.K, self.H, rng)

        for it in range(self.max_iter):
            G = self._prototipos(X, row_labels, col_labels, self.K, self.H)
            historico_custo.append(self._parada(X, G, row_labels, col_labels))

            novos_rl = self._reat_objetos(G, X, col_labels, self.K)
            novos_cl = self._reat_variaveis(G, X, novos_rl, self.H)

            if (np.array_equal(novos_rl, row_labels)
                    and np.array_equal(novos_cl, col_labels)):
                break

            row_labels, col_labels = novos_rl, novos_cl
        else:
            # não convergiu: G e W precisam corresponder aos rótulos finais
            G = self._prototipos(X, row_labels, col_labels, self.K, self.H)
            historico_custo.append(self._parada(X, G, row_labels, col_labels))

        return {
            'G': G,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'W': historico_custo[-1],
            'hist': np.array(historico_custo),
            'n_iter': len(historico_custo),
        }



    def fit(self, X):
        X = np.asarray(X, dtype=float)
        melhor, Ws = None, []

        for r in range(self.n_init):
            semente = (None if self.random_state is None
                       else self.random_state + r)
            sol = self._double_k_means(X, semente)
            Ws.append(sol['W'])
            if melhor is None or sol['W'] < melhor['W']:
                melhor = sol

        self.G_ = melhor['G']
        self.row_labels_ = melhor['row_labels']
        self.col_labels_ = melhor['col_labels']
        self.W_ = melhor['W']
        self.hist_ = melhor['hist']
        self.n_iter_ = melhor['n_iter']
        self.Ws_ = np.array(Ws)      # as n_init soluções, para medir a sensibilidade
        return self


        

def avaliar_pares(resultados, X, y=None, imprimir=True, nome_matriz=None):
    """Calcula silhueta (e métricas externas, se y for dado) para cada par.

    Preenche o dicionário in-place e devolve (K*, H*) — o par de maior silhueta.

    X : a MESMA matriz usada para gerar as partições em `resultados`
    y : rótulos verdadeiros (opcional) — habilita ARI e pureza
    nome_matriz : rótulo só para o cabeçalho (ex.: 'Xz', 'log1p'), para deixar
                registrado na saída qual matriz foi avaliada
    """
    X = np.asarray(X, dtype=float)

    for (K, H), r in resultados.items():
        rl = r["row_labels"]

        if len(rl) != len(X):
            raise ValueError(
                f"X tem {len(X)} objetos, mas a partição ({K},{H}) tem {len(rl)}"
            )

        ss = silhouette_samples(X, rl)

        r["sil"] = ss.mean()
        r["sil_por_cluster"] = [ss[rl == k].mean() for k in range(K)]
        r["frac_negativa"] = (ss < 0).mean()
        r["tamanhos"] = np.bincount(rl, minlength=K)

        if y is not None:
            # linhas = classes verdadeiras, colunas = clusters
            cm = confusion_matrix(y, rl)
            r["confusao"] = cm
            r["ari"] = adjusted_rand_score(y, rl)
            # pureza: dentro de CADA CLUSTER (coluna), a classe dominante
            r["pureza"] = cm.max(axis=0).sum() / cm.sum()

    par_estrela = max(resultados, key=lambda p: resultados[p]["sil"])

    if imprimir:
        if nome_matriz:
            print(f"matriz avaliada: {nome_matriz}\n")

        cab = f"{'(K,H)':>7} | {'Sil':>7} | {'W':>12} | {'menor':>6} | {'s<0':>6}"
        if y is not None:
            cab += f" | {'ARI':>7} | {'pureza':>7}"
        print(cab)
        print("-" * len(cab))

        for (K, H), r in resultados.items():
            linha = (f"{f'({K},{H})':>7} | {r['sil']:7.4f} | {r['W']:12.1f} | "
                    f"{r['tamanhos'].min():6d} | {r['frac_negativa']:5.1%}")
            if y is not None:
                linha += f" | {r['ari']:7.4f} | {r['pureza']:7.4f}"
            print(linha)

        K_e, H_e = par_estrela
        print(f"\nK* = {K_e}  (par ({K_e},{H_e}), "
            f"Sil = {resultados[par_estrela]['sil']:.4f})")

        if y is not None:
            maj = np.bincount(y).max() / len(y)
            print(f"   ARI do par escolhido = {resultados[par_estrela]['ari']:.4f}")
            print(f"   classe majoritária   = {maj:.4f}")

    return par_estrela