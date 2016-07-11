"""
basic exploration of data set
"""


import pandas
import time
import datetime
from collections import defaultdict
from joblib import Parallel, delayed

def save_group(name, group):
    print name
    client_sample = group #.sort_values(by=['Producto_ID', 'Semana'])
    client_sample = client_sample.merge(product, on=['Producto_ID'])
    client_sample = client_sample.merge(client, on=['Cliente_ID'])
    client_sample = client_sample.merge(town_state, on=['Agencia_ID'])
    client_sample = client_sample.sort_values(by=['NombreProducto', 'Semana'])
    client_sample.to_csv('client_data/'+str(name)+'.csv', index=False)

def extract_client(df_train):
    # train_groups = df_train.groupby(['Cliente_ID', 'Producto_ID'])
    train_groups = df_train.groupby(['Cliente_ID'])
    Parallel(n_jobs=48) (delayed (save_group) (name, group) for name, group in train_groups)

# TODO
def explore_client(cid):
    pass

if __name__=='__main__':
    print 'reading data'
    df_train = pandas.read_csv('data/train.csv')

    global product, client, town_state
    # product id, name
    product = pandas.read_csv('data/producto_tabla_stem.csv')
    # client id, name
    client = pandas.read_csv('data/cliente_tabla.csv')
    # agency id, location
    town_state = pandas.read_csv('data/town_state.csv')

    extract_client(df_train)
