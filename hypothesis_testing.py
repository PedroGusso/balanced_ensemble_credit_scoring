import Orange.evaluation
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata
import sys


def omnibus_posthoc(resultados, filename):
    # Friedman + Nemenyi
    stat, p = friedmanchisquare(resultados['Logistic Regression'], resultados['Balanced Random Forest'],
                                resultados['Balanced Bagging'], resultados['Easy Ensemble'], resultados['RUS Boost'])
    sys.stdout = open("results/friedman_nemenyi-" + filename + ".txt", "w")
    print(print('stat=%.3f, p=%.3f' % (stat, p)))
    if p > 0.05:
        print('Nao ha dif estatistica')
    else:
        print('Ha dif estatistica')

    # Ranking Data
    # * -1 porque quanto menor melhor
    ranking = (resultados * -1).apply(lambda row: rankdata([row['Logistic Regression'], row['Balanced Random Forest'],
                                                           row['Balanced Bagging'], row['Easy Ensemble'],
                                                           row['RUS Boost']]), axis=1)
    # Imprime o resultado do teste de friedman e nemenyi
    print(ranking)
    sys.stdout.close()

    names = resultados.columns.values
    avranks = ranking.mean()
    cd = Orange.evaluation.compute_CD(avranks, len(ranking), alpha="0.1")  # numero de linhas da tabela
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    plt.savefig('results/critical_distance-' + filename)

