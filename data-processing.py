import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def home_credit():
    # Experimento com a base home-credit-default-risk
    pd.set_option('display.max_rows', None)
    data = pd.read_csv("home-credit-default-risk/application_train.csv")
    # Cria um dataframe somente com as colunas categoricas e substitui os valores nulos pela moda da coluna
    categoric = data.iloc[:, [2, 3, 4, 5, 11, 12, 13, 14, 15, 28, 32, 40, 86, 87, 89, 90]].fillna(data.agg(
        lambda x: x.value_counts().index[0]))
    # Faz o One-Hot-Encoding das colunas categoricas
    categoric = pd.get_dummies(categoric)
    # Tira a primera coluna de contagem e as demais colunas categoricas para deixar um dataframe somente numerico
    cols = [0, 2, 3, 4, 5, 11, 12, 13, 14, 15, 28, 32, 40, 86, 87, 89, 90]
    data.drop(data.columns[cols], axis=1, inplace=True)
    # Subsitui os valores das colunas numericas pela media aritmetica da coluna
    data.fillna(data.mean(), inplace=True)
    # Concatena o datafram categorico com o dataframe numerico
    credit = pd.concat([data, categoric], axis=1)
    # Salva a base processada na pasta da base em formato CSV
    credit.to_csv('home-credit-default-risk/data.csv', encoding='utf-8', index=False)


def give_me_credit():
    # Experimento com a base Give Me Some Credit
    data = pd.read_csv("give-me-some-credit-dataset/cs-training.csv")
    data.fillna(data.mean(), inplace=True)
    data.drop(data.columns[0], inplace=True, axis=1)
    data.to_csv('give-me-some-credit-dataset/data.csv', encoding='utf-8', index=False)


def german():
    # Experimento com a base german
    raw_data = open('german-credit-data-data-set/german.data')
    data = np.genfromtxt(raw_data, dtype=str)
    # One hot enconding
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19])],
                                          remainder='passthrough')
    credit = np.array(columnTransformer.fit_transform(data), dtype=np.str)
    credit = credit.astype(float)
    np.savetxt('german-credit-data-data-set/data.csv', credit, delimiter=',')


if __name__ == '__main__':
    home_credit()
    give_me_credit()
    german()
