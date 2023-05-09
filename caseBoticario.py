"""
Created on Tue Apr 11 15:33:04 2023

@author: Bruna Moura Bergmann
"""


import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import datetime

pd.options.display.max_columns = None
pd.options.display.max_rows = None
sns.set_palette("gist_ncar")
sns.set_style("whitegrid")
seed = 42

import os

dir_path = os.getcwd()

def fill_categoric_field_with_value(serie):
    names = serie.unique()
    values = list(range(1, names.size + 1))
    
    #a tabela de valores continha um float(nan) mapeado para um valor inteiro. Solução foi mudar na tabela de valores colocando o None
    nan_index = np.where(pd.isna(names))
    if len(nan_index) > 0 and len(nan_index[0]) > 0:
        nan_index = nan_index[0][0]
        values[nan_index] = None
    #else:
        #print("Não encontrou nan em " + str(names))
        
    return serie.replace(names,values)

# Pegar o dia de inicio de cada ano e somar os dias de cada ciclo
def obter_data(row , df_tamanho_ciclos):
    ciclo = row['CICLO']
    ano = row['ANO']
    delta = df_tamanho_ciclos.loc[df_tamanho_ciclos['ANO'] == ano]['DELTA'].values[0]
    a_data_inicial = datetime.date(int(ano), 1, 1) + datetime.timedelta(days=(delta * (ciclo - 1)) )
    a_data_final = a_data_inicial + datetime.timedelta(days=delta)
    return a_data_inicial , a_data_final

# =============================================================================
# Importação dos dados
# =============================================================================
print("-> Carregando os dados")
path = dir_path + '/case_ds_gdem.sqlite3'

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect(path)
df = pd.read_sql_query("SELECT * from vendas", con)

print('A base possui', df.shape[0], 'linhas e', df.shape[1], 'variáveis (atributos)')

# =============================================================================
# Preparação dos dados aplicando as conclusões da exploracao
# =============================================================================

print("-> Preparação dos Dados")
df['CICLO'] = df['COD_CICLO'].astype(str).str.slice(start=4).astype(int)
df['ANO'] = df['COD_CICLO'].astype(str).str.slice(stop=4).astype(int)
df.drop(['FLG_CAMPANHA_MKT_E' , 'FLG_CAMPANHA_MKT_F'],axis = 1,inplace=True)
df.dropna(subset = ['VL_PRECO'], inplace = True ) 
df = df.drop(df[df['PCT_DESCONTO'] > 100].index)
df['DES_CATEGORIA_MATERIAL'] = fill_categoric_field_with_value(df['DES_CATEGORIA_MATERIAL'])
df['DES_MARCA_MATERIAL'] = fill_categoric_field_with_value(df['DES_MARCA_MATERIAL'])
df['PCT_DESCONTO'].fillna(0, inplace=True)

#Temporização
a = df.groupby('ANO')['CICLO'].idxmax()
df_tamanho_ciclos = df.loc[a][['ANO', 'CICLO']]

#assumption, 2021 tera 17 ciclos
df_tamanho_ciclos['CICLO'] = df_tamanho_ciclos.apply(lambda x : 17 if x['ANO'] == 2021 else x['CICLO'] , axis = 1)
df_tamanho_ciclos['DELTA'] = 365 / df_tamanho_ciclos['CICLO']

df[['DATA_INICIO' , 'DATA_FIM']] = df.apply(obter_data , df_tamanho_ciclos = df_tamanho_ciclos , result_type='expand' , axis=1)
df.sort_values(by='DATA_INICIO',inplace=True)

df['DATA_INICIO'] = df['DATA_INICIO'].apply(lambda x: datetime.datetime.combine(x, datetime.datetime.min.time())).apply(lambda x: x.timestamp())
df['DATA_FIM'] = df['DATA_FIM'].apply(lambda x: datetime.datetime.combine(x, datetime.datetime.min.time())).apply(lambda x: x.timestamp())


# =============================================================================
# Preparação das bases de treino e de predição
# =============================================================================
print("-> Preparação das bases de treino e teste")

df_base=df.dropna(subset=['QT_VENDA'], inplace=False)

df_a_prever = df[df['QT_VENDA'].isnull()]
df_a_prever = df_a_prever.drop('QT_VENDA' , axis=1)

# Separação teste e treino

# divide a base em partições de treino (70%) e teste (30%) com dados estratificados pela variável alvo
X_train, X_test, y_train, y_test = train_test_split(
    df_base.drop(['QT_VENDA'], axis=1), 
    df_base['QT_VENDA'],
    test_size=0.3,
    stratify = df_base[['ANO','CICLO']])

# =============================================================================
# Normalização dos valores
# =============================================================================

print("-> Normalização dos dados")

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test =scaler.transform(X_test)

# =============================================================================
# Criação do Modelo
# =============================================================================
random_forest = RandomForestRegressor(random_state=0, max_depth = 33, n_jobs=4)

print("-> Treinamento do Modelo")

random_forest.fit(X_train,y_train)


#salvar o modelo
import pickle
# salvando o modelo final
pkl_filename = dir_path + "/modelo_dump.pkl"
print("-> Salvando o Modelo no arquivo " + pkl_filename)
with open(pkl_filename, 'wb') as file:
    pickle.dump(random_forest, file)

# =============================================================================
# Aplicação do Modelo para Previsão
# =============================================================================
print("-> Aplicação do Modelo para Previsão")

df_a_prever_scaled = scaler.transform(df_a_prever)

resultado = random_forest.predict(df_a_prever_scaled)
resultado = resultado.astype('int')

df_a_prever['QT_VENDA'] = resultado



print("-> Salvando o resultado da predição no arquivo " + dir_path + "/resultado.xlsx")
df_a_prever.to_excel(dir_path + "/resultado.xlsx",sheet_name='Previsao') 



