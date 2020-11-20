import pandas as pd
import numpy as np

# imprimir strings muito compridas, até 90 caracteres
pd.set_option('display.max_colwidth', 90)
# imprimir todas as colunas
pd.set_option('display.max_rows', 500)


def give_credit():
    # Desbalanceamento do dataset give me some credit
    data = pd.read_csv("give-me-some-credit-dataset/cs-training.csv")
    # Para ver o número de valores nulos por coluna
    print("Number of empty values per column")
    print(data.isna().sum())
    data1 = data['SeriousDlqin2yrs'].count()
    data2 = data.groupby('SeriousDlqin2yrs').size()
    data3 = data2.groupby(level=0).apply(lambda x: 100 * x / data1)
    data4 = pd.merge(data2.rename('count'), data3.rename('percentage'), how='left', on='SeriousDlqin2yrs')
    print("\nClass distribution")
    print(data4)


def taiwanese():
    # Desbalanceamento do taiwanese credit data set
    data = pd.read_excel("taiwanese-credit-data-set/default of credit card clients.xls", usecols='Y').iloc[1:]
    # Para ver o número de valores nulos por coluna
    print("Number of empty values per column")
    print(data.isna().sum())
    data1 = data['Y'].count()
    data2 = data.groupby('Y').size()
    data3 = data2.groupby(level=0).apply(lambda x: 100 * x / data1)
    data4 = pd.merge(data2.rename('count'), data3.rename('percentage'), how='left', on='Y')
    print("\nClass distribution")
    print(data4)


def german():
    # Desbalanceamento do german credit data dataset
    raw_data = open('german-credit-data-data-set/german.data')
    credit = np.genfromtxt(raw_data)
    total = credit.shape
    data = pd.DataFrame(credit)
    data2 = data.groupby(by=20).size()
    data3 = data2.groupby(level=0).apply(lambda x: 100*x / total[0])
    data4 = pd.concat([data2, data3.reindex(data2.index)], axis=1)
    print("\nClass distribution")
    print(data4)


def australian():
    # Desbalanceamento do australian credit approval data set
    raw_data = open('australian-credit-approval-data-set/australian.dat')
    credit = np.genfromtxt(raw_data)
    total = credit.shape
    data = pd.DataFrame(credit)
    data2 = data.groupby(by=14).size()
    data3 = data2.groupby(level=0).apply(lambda x: 100*x / total[0])
    data4 = pd.concat([data2, data3.reindex(data2.index)], axis=1)
    print(data4)


def home_credit():
    # Desbalanceamento do home credit default risk
    data = pd.read_csv("home-credit-default-risk/application_train.csv")
    data1 = data['TARGET'].count()
    data2 = data.groupby('TARGET').size()
    data3 = data2.groupby(level=0).apply(lambda x: 100*x / data1)
    data4 = pd.merge(data2.rename('count'), data3.rename('percentage'), how='left', on='TARGET')
    print(data4)


if __name__ == '__main__':
    print("Give Me Some Credit\n")
    give_credit()
    print("\n---------------")
    print("Taiwanese\n")
    taiwanese()
    print("\n---------------")
    print("German\n")
    german()
    print("\n---------------")
    print("Australian\n")
    australian()
    print("---------------")
    print("Home Credit Default Risk\n")
    home_credit()
