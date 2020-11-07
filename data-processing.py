import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def lending_club():
    # Experimento com a base Lending Club
    pd.options.display.width = 0
    pd.set_option('display.max_rows', None)
    # Carrega a base com um sample com as 300.000 instancias mais recentes
    data = pd.read_csv("lending-club-loan-data/loan.csv", nrows=300000)
    # Retira as colunas com taxa de nulos maior que 80%
    data.drop(['id', 'member_id', 'url',
               'orig_projected_additional_accrued_interest', 'hardship_type', 'hardship_reason', 'hardship_status',
               'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
               'payment_plan_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
               'hardship_payoff_balance_amount', 'hardship_last_payment_amount', 'debt_settlement_flag_date',
               'settlement_status', 'settlement_date', 'settlement_amount', 'settlement_percentage',
               'settlement_term', 'sec_app_mths_since_last_major_derog', 'sec_app_revol_util', 'revol_bal_joint',
               'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
               'sec_app_open_act_il', 'sec_app_num_rev_accts', 'sec_app_chargeoff_within_12_mths',
               'sec_app_collections_12_mths_ex_med', 'verification_status_joint', 'dti_joint', 'annual_inc_joint',
               'desc', 'mths_since_last_record'], axis=1, inplace=True)
    # Retira as categorias de target invalidas e transforma as validas em grupos de 1 e 0
    invalid_target = ['Default', 'Does not meet the credit policy. Status:Charged Off',
                      'Does not meet the credit policy. Status:Fully Paid', 'In Grace Period', 'Late (16-30 days)',
                      'Late (31-120 days)']
    data.drop(data[(data['loan_status'] == invalid_target[0]) | (data['loan_status'] == invalid_target[1]) |
                   (data['loan_status'] == invalid_target[2]) | (data['loan_status'] == invalid_target[3]) |
                   (data['loan_status'] == invalid_target[4]) | (data['loan_status'] == invalid_target[5])].index,
              inplace=True)
    data['loan_status'].replace(['Charged Off'], 1, inplace=True)
    data['loan_status'].replace(['Current', 'Fully Paid'], 0, inplace=True)
    # Retira colunas categoricas com muitos valores unicos para nao degradar a performance do one hot encoding
    data.drop(['addr_state', 'last_pymnt_d', 'issue_d', 'last_credit_pull_d', 'earliest_cr_line', 'zip_code', 'title',
               'emp_title'], axis=1, inplace=True)
    # Cria um dataframe somente com as colunas categoricas e substitui os valores nulos pela moda da coluna
    categoric = data.iloc[:, [3, 6, 7, 8, 9, 11, 13, 14, 24, 35, 39, 95, 96, 97]].fillna(
        data.agg(lambda x: x.value_counts().index[0]))
    # Faz o One-Hot-Encoding das colunas categoricas
    categoric = pd.get_dummies(categoric)
    # Tira as colunas categoricas para deixar um dataframe somente numerico
    cols = [3, 6, 7, 8, 9, 11, 13, 14, 24, 35, 39, 95, 96, 97]
    data.drop(data.columns[cols], axis=1, inplace=True)
    # Subsitui os valores das colunas numericas pela media aritmetica da coluna
    data.fillna(data.mean(), inplace=True)
    # Concatena o dataframe categorico com o dataframe numerico
    credit = pd.concat([data, categoric], axis=1)
    # Coloca a coluna do target na primeira posicao para facilitar na analise
    target = data.iloc[:, [6]]
    credit.drop(credit.columns[[6]], axis=1, inplace=True)
    final = pd.concat([target, credit], axis=1)
    # Salva a base processada na pasta da base em formato CSV
    final.to_csv('lending-club-loan-data/data.csv', encoding='utf-8', index=False)


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
    lending_club()
    home_credit()
    give_me_credit()
    german()

