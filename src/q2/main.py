from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pandas as pd
from models.logistic_reg import ajustar_logistica
from models.gauss_bayes import BayesGaussianoMV, ajustar_gaussiano
from sklearn.model_selection import StratifiedKFold
from utils.k_folds import ext_kfold, int_kfold, metricas, resumir
from sklearn.linear_model import LogisticRegression
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

# rodando 30 x 10 folds
n_rep = 30
res_log, hiper_log = ext_kfold(X, y, GRADE_LOG, MODELO_LOG, nome='regressão logística', n_rep=n_rep)
res_gauss, hiper_gauss = ext_kfold(X, y, GRADE_GAUSS, MODELO_GAUSS, nome='bayesiano gaussiano', n_rep=n_rep)

# resultados
resumir(res_log, hiper_log, 'regressão logística')
resumir(res_gauss, hiper_gauss, 'bayesiano gaussiano')

#salvando
RESULTADOS = Path(__file__).resolve().parent / 'results'
RESULTADOS.mkdir(exist_ok=True)

np.save(RESULTADOS / f'res_log_{n_rep}.npy', res_log)
np.save(RESULTADOS / f'res_gauss_{n_rep}.npy', res_gauss)

with open(RESULTADOS / f'hiper_log_{n_rep}.pkl', 'wb') as arq:
    pickle.dump(hiper_log, arq)
with open(RESULTADOS / f'hiper_gauss_{n_rep}.pkl', 'wb') as arq:
    pickle.dump(hiper_gauss, arq)

# abrindo
# res_log = np.load(RESULTADOS / 'res_log_30.npy')
# with open(RESULTADOS / 'hiper_log_30.pkl', 'rb') as arq:
#     hiper_log = pickle.load(arq)
# resumir(res_log, hiper_log, 'regressão logística')

# res_gauss = np.load(RESULTADOS / 'res_gauss_30.npy')
# with open(RESULTADOS / 'hiper_gauss_30.pkl', 'rb') as arq:
#     hiper_gauss = pickle.load(arq)
# resumir(res_gauss, hiper_gauss, 'bayesiano gaussiano')