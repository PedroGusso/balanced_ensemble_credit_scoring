import pandas as pd
import numpy as np
from psi import calculate_psi
from scipy.stats import ks_2samp

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import RUSBoostClassifier

from hypothesis_testing import omnibus_posthoc

# Classificadores

# Logistic Regression
# E preciso aumentar o numero de iteracoes conforme a base do contrario da erro
lr = LogisticRegression(random_state=100, n_jobs=-1, max_iter=10000, class_weight='balanced')

# Balanced Random Forest
brf = BalancedRandomForestClassifier(random_state=100, n_jobs=-1)

# Balanced Bagging
bbc = BalancedBaggingClassifier(random_state=100, n_jobs=-1)

# Easy Ensemble
eac = EasyEnsembleClassifier(random_state=100, n_jobs=-1)

# RUS Boost
rbc = RUSBoostClassifier(random_state=100)

# Tabela com as estatisticas
table = []

# Nomes e metodos dos classificadores
names = ['Logistic Regression', 'Balanced Random Forest', 'Balanced Bagging', 'Easy Ensemble', 'RUS Boost']
methods = [lr, brf, bbc, eac, rbc]

# Matrizes para fazer os testes de friedman e nemenyi
friedman_nemenyi_ks = pd.DataFrame(columns=names)
friedman_nemenyi_auc = pd.DataFrame(columns=names)
friedman_nemenyi_f1 = pd.DataFrame(columns=names)
friedman_nemenyi_psi = pd.DataFrame(columns=names)


def home_credit():
    # Experimento com a base home-credit-default-risk
    data = pd.read_csv("home-credit-default-risk/data.csv")
    credit = data.to_numpy()
    X, y = credit[:, 1:], credit[:, :1]
    dataset_label = 1
    return X, y, dataset_label


def give_credit():
    # Experimento com a base Give Me Some Credit
    data = pd.read_csv("give-me-some-credit-dataset/data.csv")
    credit = data.to_numpy()
    X, y = credit[:, 1:], credit[:, :1]
    dataset_label = 1
    return X, y, dataset_label


def taiwanese():
    # Experimento com a base taiwanese
    data = pd.read_excel("taiwanese-credit-data-set/default of credit card clients.xls", header=1)
    credit = data.to_numpy()
    X, y = credit[:, 1:-1], credit[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def australian():
    # Experimento com a base australian
    data = open('australian-credit-approval-data-set/australian.dat')
    credit = np.genfromtxt(data)
    X, y = credit[:, :-1], credit[:, -1:]
    dataset_label = 1
    return X, y, dataset_label


def german():
    # Experimento com a base german
    data = pd.read_csv('german-credit-data-data-set/data.csv', header=None)
    credit = data.to_numpy()
    X, y = credit[:, :-1], credit[:, -1:]
    dataset_label = 2
    return X, y, dataset_label


def stats(X, y, dataset_label, dataset_name):
    # Experimento

    # Lista de resultados do KS, AUC, F1 e PSI para inserir no dataframe que sera utilizado para
    # calcular os testes de friedman e nemenyi
    ks_list = []
    auc_list = []
    f1_list = []
    psi_list = []

    # Lista com as plotagens do detection rate
    dr_list_plot = []

    # Lista de parametros para o grid search
    parameters = {
        'n_estimators': [50, 100, 200],
        'replacement': [True, False]
    }

    for method, name in zip(methods, names):

        # Hold out 70% para treinamento e 30% para teste, estratificada
        # Nao e possivel fazer cross validation por causa do PSI pois precisa dos resultados do treinamento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=100, stratify=y)

        def ks_stat(y_target, y_proba):
            return ks_2samp(y_proba[y_target == dataset_label], y_proba[y_target != dataset_label])[0]

        ks_scorer = make_scorer(ks_stat, needs_proba=True, greater_is_better=True)

        # Grid search para todos menos o de regressao logistica
        if method != lr:
            clf = GridSearchCV(method, parameters, scoring={'ks': ks_scorer}, refit='ks')
            clf.fit(X_train, np.ravel(y_train, order='C'))
            y_proba_train = clf.predict_proba(X_train)[:, 1]
            y_pred_method = clf.predict(X_test)
            y_proba_method = clf.predict_proba(X_test)

        else:
            method.fit(X_train, np.ravel(y_train, order='C'))
            y_proba_train = method.predict_proba(X_train)[:, 1]
            y_pred_method = method.predict(X_test)
            y_proba_method = method.predict_proba(X_test)

        fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_test, y_proba_method[:, 1], pos_label=dataset_label)
        plt.plot(fpr_roc, tpr_roc)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_method).ravel()
        fpr = fp/(fp+tn)
        fnr = fn/(fn+tp)

        auc = roc_auc_score(y_test, y_proba_method[:, 1])
        f1 = f1_score(y_test, y_pred_method, average=None)

        # ks com as probabilidades da classe positiva e a negativa
        ks = ks_2samp(y_proba_method[:, 1][y_test[:, 0] == dataset_label],
                      y_proba_method[:, 1][y_test[:, 0] != dataset_label])
        ks_result = ks[0]
        ks_pvalue = ks[1]

        y_pred_method_flipped = np.flip(y_pred_method)
        y_proba_method_flipped = np.flip(y_proba_method[:, 1])

        # Nao informado. Substituir y_test pelas probabilidades do treino
        # Fazer um histograma das duas variaveis passadas como parametro
        psi = calculate_psi(y_proba_train, y_proba_method_flipped, axis=1)

        # Ver se nao esta invertido, pegar o complemento do KS, 1 - o valor de KS
        # Adiciona o ks, auc, f1 score e psi atual na lista para fazer o teste omnibus e post hoc
        ks_list.append(ks_result)
        auc_list.append(auc)
        f1_list.append(f1[1])
        psi_list.append(psi)

        # Tabela de resultados estatisticos
        # Apenas pego o f1 score da classe positiva
        table.append([dataset, name, fpr, fnr, auc, f1[1], ks_result, ks_pvalue, psi])

        # Fazer o detection rate (sensitivity) da seguinte forma: Ordenar uma matrix com 3 colunas (proba, pred)
        # Ordernar por proba e dividir essa matriz de 10 % em 10% fazendo um ponto de detection rate para cada divisao
        # Plotar o grafico com todos os pontos

        y_proba_method_percent = [x * 100 for x in y_proba_method[:, 1]]
        dr_matrix = np.array((y_proba_method_percent, y_pred_method_flipped))
        dr_matrix = np.transpose(dr_matrix)
        dr_matrix = np.flip(np.sort(dr_matrix, axis=0), axis=0)
        dr_groups = np.array_split(dr_matrix, 10, axis=0)
        dr_list = []
        cumulative_dr = 0
        for i in range(len(dr_groups)):
            # todos os positivos do subgrupo dividido pelo numero total de positivos da base inteira (cumulativo)
            # Como o DR e sobre todos os positivos, eu so dividi por todos os positivos
            # Bloco try except caso tenha divisao por zero
            try:
                cumulative_dr += len([x for x in dr_groups[i][:, 1] if x == dataset_label]) / len(
                    [i for i in dr_matrix[:, 1] if i == dataset_label])
            except:
                cumulative_dr += 0
            dr_list.append(cumulative_dr)
        dr_list_plot.append(dr_list)

    # Plota o grafico ROC
    plt.title('Receiver Operating Characteristic: ' + dataset_name)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(names)
    plt.savefig('results/ROC-' + dataset)
    plt.clf()

    # Plota o grafico DR
    plt.title('Cumulative Detection Rate X Test Subgroups: ' + dataset_name)
    plt.xlabel('Test Subgroups')
    plt.ylabel('Cumulative Detection Rate')
    for p, name in zip(dr_list_plot, names):
        plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], p, marker='o')
    plt.grid(True)
    plt.xticks(np.arange(1, 11, step=1))
    plt.legend(names)
    plt.savefig('results/DR-' + dataset)
    plt.clf()

    # Coloca as listas de KS, AUC, F1 e PSI nos dataframes
    friedman_nemenyi_ks.loc[len(friedman_nemenyi_ks)] = ks_list
    friedman_nemenyi_auc.loc[len(friedman_nemenyi_auc)] = auc_list
    friedman_nemenyi_f1.loc[len(friedman_nemenyi_f1)] = f1_list
    friedman_nemenyi_psi.loc[len(friedman_nemenyi_psi)] = psi_list


if __name__ == '__main__':
    # Header do dataframe que sera salvo em arquivo com as estatisticas
    header = ['Dataset', 'Classifiers', 'FPR', 'FNR', 'AUC', 'F1 Score', 'KS Result', 'KS p-value', 'PSI']
    datasets_dict = {
        "Australian": australian(),
        "German": german(),
        "Taiwanese": taiwanese(),
        "Give Me Some Credit": give_credit(),
        "home-credit-default-risk": home_credit(),
    }
    # tqdm no looping para mostrar a barra de progresso do algoritmo
    for dataset in tqdm(datasets_dict):
        dataset_X, dataset_y, target_label = datasets_dict[dataset]
        stats(dataset_X, dataset_y, target_label, dataset)
    stats_table = pd.DataFrame(data=table, columns=header)
    stats_table.to_csv("results/metrics.csv")
    omnibus_posthoc(friedman_nemenyi_ks, "ks")
    omnibus_posthoc(friedman_nemenyi_auc, "auc")
    omnibus_posthoc(friedman_nemenyi_f1, "f1")
    omnibus_posthoc(friedman_nemenyi_psi, "psi")
