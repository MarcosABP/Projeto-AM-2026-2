import json
import pickle
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')                   
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from models.gauss_bayes import BayesGaussianoMV
from models.knn_bayes import BayesKNN
from models.parzen_bayes import BayesParzen

from models.maj_vote import voto_dos_folds
from utils.k_folds import ext_kfold
from utils.metrics import tabela_comparativa, resumir, testar_todas
from utils.curve import curva_aprendizagem
import utils.plots as plots

warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# configuração
# ---------------------------------------------------------------------------

VERSAO = 'kestrela'     # '2classes' (variável resposta original) 'kestrela' os K* grupos de objetos da Questão 1
RODAR = False           # True: roda os 30 x 10 folds. False: carrega de results/
RODAR_CURVA = False     # True: recalcula a curva de aprendizagem (~4 min)
N_REP = 30

RAIZ = Path(__file__).resolve().parents[2]          # pasta AM-1
DADOS = RAIZ / 'data'
RESULTADOS = Path(__file__).resolve().parent / 'results'
RESULTADOS.mkdir(exist_ok=True)

# a versão de 2 classes mantém os nomes de arquivo já existentes
SUF = '' if VERSAO == '2classes' else f'_{VERSAO}'
plots.SUFIXO = SUF


# ---------------------------------------------------------------------------
# dados
# ---------------------------------------------------------------------------

nomes = []
with open(DADOS / 'spambase.names') as arq:
    for linha in arq:
        linha = linha.strip()
        if linha.endswith('continuous.'):
            nomes.append(linha.split(':')[0].strip())

nomes.append('spam')

spam_data = pd.read_csv(DADOS / 'spambase.data', header=None, names=nomes)

X = spam_data.iloc[:, :-1].values

if VERSAO == '2classes':
    y = spam_data.iloc[:, -1].values
elif VERSAO == 'kestrela':
    pasta_q1 = RAIZ / 'src' / 'q1' / 'results'
    y = np.load(pasta_q1 / 'rotulos_kestrela.npy')
    with open(pasta_q1 / 'kestrela.json') as arq:
        info = json.load(arq)
    assert len(y) == len(X), 'rótulos do K* não correspondem à base'
    print(f'versão K*: K* = {info["K"]} (par ({info["K"]},{info["H"]})), '
          f'tamanhos dos grupos = {info["tamanhos"]}')
else:
    raise ValueError(f'VERSAO desconhecida: {VERSAO}')


# ---------------------------------------------------------------------------
# a) classificadores e grades de hiperparâmetros
# ---------------------------------------------------------------------------

# REGRESSÃO LOGÍSTICA
GRADE_LOG = ([{'penalty': 'l2', 'C': c} for c in (0.01, 1.0, 5.0, 10.0, 50.0, 100.0)]
             + [{'penalty': None}])
MODELO_LOG = lambda g: LogisticRegression(solver='lbfgs', max_iter=20000,
                                          random_state=42, **g)

# BAYESIANO GAUSSIANO
GRADE_GAUSS = [{'gamma': v} for v in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)]
MODELO_GAUSS = lambda g: BayesGaussianoMV(**g)

# KNN BAYESIANO
GRADE_KNN = [{'k': k, 'metric': m}
             for m in ('euclidean', 'cityblock', 'chebyshev')
             for k in (1, 3, 5, 7, 11, 15, 21, 31)]
MODELO_KNN = lambda g: BayesKNN(**g)

# JANELA DE PARZEN
GRADE_PARZEN = [{'h': h} for h in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)]
MODELO_PARZEN = lambda g: BayesParzen(**g)

CLASSIFICADORES = {
    'log':    ('regressão logística', GRADE_LOG,    MODELO_LOG),
    'gauss':  ('bayesiano gaussiano', GRADE_GAUSS,  MODELO_GAUSS),
    'parzen': ('bayesiano parzen',    GRADE_PARZEN, MODELO_PARZEN),
    'knn':    ('knn bayesiano',       GRADE_KNN,    MODELO_KNN),
}


# ---------------------------------------------------------------------------
# a) 30 x 10-folds
# ---------------------------------------------------------------------------

def rodar_e_salvar(sigla, nome, grade, criar_modelo, n_rep):
    res, hiper, preds, ytes = ext_kfold(X, y, grade, criar_modelo,
                                        nome=nome, n_rep=n_rep)
    np.save(RESULTADOS / f'res_{sigla}{SUF}_{n_rep}.npy', res)
    with open(RESULTADOS / f'extra_{sigla}{SUF}_{n_rep}.pkl', 'wb') as arq:
        pickle.dump({'hiper': hiper, 'preds': preds, 'ytes': ytes}, arq)
    resumir(res, hiper, nome)
    return res, hiper, preds, ytes


def carregar(sigla, n_rep):
    res = np.load(RESULTADOS / f'res_{sigla}{SUF}_{n_rep}.npy')
    with open(RESULTADOS / f'extra_{sigla}{SUF}_{n_rep}.pkl', 'rb') as arq:
        e = pickle.load(arq)
    return res, e['hiper'], e['preds'], e['ytes']


if RODAR:
    saida = {s: rodar_e_salvar(s, nome, grade, modelo, N_REP)
             for s, (nome, grade, modelo) in CLASSIFICADORES.items()}
else:
    saida = {s: carregar(s, N_REP) for s in CLASSIFICADORES}
    for s, (nome, _, _) in CLASSIFICADORES.items():
        resumir(saida[s][0], saida[s][1], nome)

# VOTO MAJORITÁRIO — combina as predições dos 4 nos mesmos folds
extras = {s: {'preds': saida[s][2], 'ytes': saida[s][3]} for s in saida}
res_voto, empates = voto_dos_folds(extras, y)
np.save(RESULTADOS / f'res_voto{SUF}_{N_REP}.npy', res_voto)
resumir(res_voto, nome='voto majoritário')
print(f'empates decididos pela priori: {empates.sum()} '
      f'({empates.sum() / (N_REP * len(y)):.1%} dos exemplos)')

res_todos = {CLASSIFICADORES[s][0]: saida[s][0] for s in saida}
res_todos['voto majoritário'] = res_voto
hipers = {CLASSIFICADORES[s][0]: saida[s][1] for s in saida}


# ---------------------------------------------------------------------------
# b) estimativa pontual e intervalo de confiança
# ---------------------------------------------------------------------------

tabela_comparativa(res_todos)
tabela_comparativa(res_todos, corrigido=True)


# ---------------------------------------------------------------------------
# c) Friedman + Nemenyi
# ---------------------------------------------------------------------------

testes = testar_todas(res_todos)

plots.plot_distribuicao_folds(res_todos)   # a)
plots.plot_hiperparametros(hipers)         # a)
plots.plot_estimativas_ic(res_todos)       # b)
plots.plot_friedman(res_todos, testes)     # c) Friedman
plots.plot_nemenyi(testes)                 # c) Nemenyi


# ---------------------------------------------------------------------------
# d) curva de aprendizagem
# ---------------------------------------------------------------------------

arq_curva = RESULTADOS / f'curva_aprendizagem{SUF}.pkl'
if RODAR_CURVA or not arq_curva.exists():
    curva = curva_aprendizagem(X, y, CLASSIFICADORES)
    with open(arq_curva, 'wb') as arq:
        pickle.dump(curva, arq)
else:
    with open(arq_curva, 'rb') as arq:
        curva = pickle.load(arq)

plots.plot_curva_aprendizagem(curva)          # F-measure: treino x teste
plots.plot_curva_teste_comparada(curva)