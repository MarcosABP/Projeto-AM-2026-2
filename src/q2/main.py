from pathlib import Path
import numpy as np
import pandas as pd
from models.gauss_bayes import BayesGaussianoMV
from models.knn_bayes import BayesKNN
from models.parzen_bayes import BayesParzen
from sklearn.linear_model import LogisticRegression

from utils.k_folds import ext_kfold
from utils.metrics import tabela_comparativa, resumir
import warnings
import pickle


warnings.filterwarnings('ignore', category=FutureWarning)

DADOS = Path('C:/AM-1/data')         # barra normal funciona no Windows

nomes = []
with open(DADOS / 'spambase.names') as arq:
    for linha in arq:
        linha = linha.strip()
        if linha.endswith('continuous.'):
            nomes.append(linha.split(':')[0].strip())

nomes.append('spam')

spam_data = pd.read_csv(DADOS / 'spambase.data', header=None, names=nomes)

X = spam_data.iloc[:, :-1].values
y = spam_data.iloc[:, -1].values





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



RESULTADOS = Path(__file__).resolve().parent / 'results'
RESULTADOS.mkdir(exist_ok=True)

# rodando 30 x 10 folds
def rodar_e_salvar(sigla, nome, grade, criar_modelo, n_rep):
    res, hiper, preds, ytes = ext_kfold(X, y, grade, criar_modelo,
                                        nome=nome, n_rep=n_rep)
    np.save(RESULTADOS / f'res_{sigla}_{n_rep}.npy', res)
    with open(RESULTADOS / f'extra_{sigla}_{n_rep}.pkl', 'wb') as arq:
        pickle.dump({'hiper': hiper, 'preds': preds, 'ytes': ytes}, arq)
    resumir(res, hiper, nome)
    return res, hiper, preds, ytes


CLASSIFICADORES = {
    'log':    ('regressão logística', GRADE_LOG,    MODELO_LOG),
    'gauss':  ('bayesiano gaussiano', GRADE_GAUSS,  MODELO_GAUSS),
    'parzen': ('bayesiano parzen',    GRADE_PARZEN, MODELO_PARZEN),
    'knn':    ('knn bayesiano',       GRADE_KNN,    MODELO_KNN),
}

n_rep = 1
saida = {sigla: rodar_e_salvar(sigla, nome, grade, modelo, n_rep)
         for sigla, (nome, grade, modelo) in CLASSIFICADORES.items()}

res_todos = {CLASSIFICADORES[s][0]: saida[s][0] for s in saida}
tabela_comparativa(res_todos)
tabela_comparativa(res_todos, corrigido=True)

# abrindo
# res_log = np.load(RESULTADOS / 'res_log_30.npy')
# with open(RESULTADOS / 'extra_log_30.pkl', 'rb') as arq:
#     extra_log = pickle.load(arq)
# resumir(res_log, extra_log['hiper'], 'regressão logística')

