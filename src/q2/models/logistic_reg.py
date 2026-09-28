
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


def ajustar_logistica(X, y, cs, grade, cv=5, seed=0, imprimir=True, solver='lbfgs'):

    acertos = np.zeros((cv, len(grade)))          # folds x configurações
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)

    for i_fold, (tr, va) in enumerate(skf.split(X, y)):
      print(f'Rodando fold {i_fold+1}/{cv} interno')

      for i, g in enumerate(grade):

        clf = LogisticRegression(solver=solver, max_iter=20000, random_state=42,**g).fit(X[tr], y[tr])
        acertos[i_fold, i] = (clf.predict(X[va]) == y[va]).mean()

    melhor = int(acertos.sum(axis=0).argmax())


    if imprimir:
      rotulos = [f"C={g['C']}" if g['penalty'] else 'None' for g in grade]
      print(' ' * 8 + ''.join(f'{r:>9}' for r in rotulos))
      for f in range(cv):
          print(f'fold {f+1:<3}' + ''.join(f'{a:>9.4f}' for a in acertos[f]))
      print(' ' * 8 + '-' * (9 * len(grade)))
      print('média  ' + ''.join(f'{a:>9.4f}' for a in acertos.mean(axis=0)))
      print(' ' * (8 + 9 * melhor + 4) + '^')
      print(f'melhor: {grade[melhor]}')

    return grade[melhor]





