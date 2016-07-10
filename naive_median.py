"""
use specified group mean/median to predict the demand in the future
"""

import time
import sys
import gc
import datetime
from collections import defaultdict
from joblib import Parallel, delayed
import pandas
import numpy as np
import ConfigParser
from sklearn.metrics import mean_squared_error as MSE

### https://www.kaggle.com/paulorzp/grupo-bimbo-inventory-demand/mean-median-lb-0-48/code
def group_median_predict_pid(df_train, df_test):
    start_time = time.time()
    all_median = df_train['Demanda_uni_equil'].median()

    product_mean = df_train.groupby(['Producto_ID','Agencia_ID'])['log_Demanda_uni_equil'].mean()
    product_mean = product_mean.apply(lambda x:np.exp(x)*0.5793)

    product_agent_mean = df_train.groupby(['Producto_ID','Agencia_ID','Cliente_ID'])\
            ['log_Demanda_uni_equil'].mean()
    product_agent_mean = product_agent_mean.apply(lambda x:np.exp(x)-0.91)
    print 'median calculating time=', time.time()-start_time

    start_time = time.time()
    df_test['Demanda_uni_equil'] = all_median
    df_test['Demanda_uni_equil'] = df_test.apply(\
            lambda x : predict_demand(x, product_agent_mean, product_mean, all_median), axis=1)
    df_test.to_csv('output/product_agent_client_group_mean_'+ \
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
            columns=['id','Demanda_uni_equil'], index=False)
    print 'predicting time=', time.time()-start_time

def predict_demand_pname(x, product_agent_mean, product_mean, all_median):
    # pid, aid, cid = x[['short_name','Agencia_ID','Cliente_ID']]
    pid, aid, cid = x[0], x[1], x[2]
    if (pid, aid, cid) in product_agent_mean.index:
        return product_agent_mean[(pid, aid, cid)]
    elif (pid, aid) in product_mean.index:
        return product_mean[(pid, aid)]
    else:
        return all_median

## https://www.kaggle.com/paulorzp/grupo-bimbo-inventory-demand/mean-median-lb-0-48/code
## use product short name
def group_median_predict_pname(df_train, df_test):
    start_time = time.time()
    all_median = df_train['Demanda_uni_equil'].median()

    product_mean = df_train.groupby(['short_name','Agencia_ID'])['log_Demanda_uni_equil'].mean()
    product_mean = product_mean.apply(lambda x:np.exp(x)*0.5793)

    product_agent_mean = df_train.groupby(['short_name','Agencia_ID','Cliente_ID'])\
            ['log_Demanda_uni_equil'].mean()
    product_agent_mean = product_agent_mean.apply(lambda x:np.exp(x)-0.91)
    print 'median calculating time=', time.time()-start_time

    start_time = time.time()
    df_test['Demanda_uni_equil'] = all_median

    # pandas dataframe apply is slow for create series for each row
    df_test['Demanda_uni_equil'] = np.apply_along_axis( \
            (lambda x : predict_demand_pname(x, product_agent_mean, product_mean, all_median)),\
            1, df_test[['short_name','Agencia_ID','Cliente_ID']].values)
    df_test.to_csv('output/pname_agent_client_group_mean_'+ \
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
            columns=['id','Demanda_uni_equil'], index=False)
    print 'predicting time=', time.time()-start_time

def log_means_predict_demand(x, labels1_mean, labels2_mean, labels3_mean, all_mean, \
        labels1_idx, labels2_idx, labels3_idx, mult_factor, plus_factor):
    x_l1 = tuple(x[labels1_idx])
    x_l2 = tuple(x[labels2_idx])
    x_l3 = tuple(x[labels3_idx])

    # print x_l1, ';', x_l2, ';', x_l3
    pred = 0
    if x_l1 in labels1_mean.index:
        pred = labels1_mean[x_l1]
    elif x_l2 in labels2_mean.index:
        pred = labels2_mean[x_l2]
    elif x_l3 in labels3_mean.index:
        pred = labels3_mean[x_l3]
    else:
        pred = all_mean

    return np.expm1(pred)*mult_factor+plus_factor

## https://www.kaggle.com/apapiu/grupo-bimbo-inventory-demand/log-means/code
def group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels1 = config.get(config_name, 'labels1').split(',')
    labels2 = config.get(config_name, 'labels2').split(',')
    labels3 = config.get(config_name, 'labels3').split(',')
    labels = list(set(labels1 + labels2 + labels3))
    print labels1, ';', labels2, ';', labels3, ';', labels
    labels1_idx = [labels.index(i) for i in labels1]
    labels2_idx = [labels.index(i) for i in labels2]
    labels3_idx = [labels.index(i) for i in labels3]
    print labels1_idx, ';', labels2_idx, ';', labels3_idx

    mult_factor = config.getfloat(config_name, 'mult_factor')
    plus_factor = config.getfloat(config_name, 'plus_factor')
    print 'mult_factor=', mult_factor, ' plus_factor=', plus_factor

    start_time = time.time()
    all_mean = df_train['log_Demanda_uni_equil'].mean()
    labels1_mean = df_train.groupby(by=labels1)['log_Demanda_uni_equil'].mean()
    labels2_mean = df_train.groupby(by=labels2)['log_Demanda_uni_equil'].mean()
    labels3_mean = df_train.groupby(by=labels3)['log_Demanda_uni_equil'].mean()

    print 'mean calculating time=', time.time()-start_time

    start_time = time.time()
    if not vali:
        df_test['Demanda_uni_equil']=np.apply_along_axis((lambda x:log_means_predict_demand(x, \
            labels1_mean, labels2_mean, labels3_mean, all_mean, \
            labels1_idx, labels2_idx, labels3_idx, mult_factor, plus_factor)), \
            1, df_test[labels].values)
        df_test.to_csv('output/'+'_'.join(labels)+'_'+config_name+'_group_log_mean_'+ \
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['id','Demanda_uni_equil'], index=False)
    else:
        global pred_demand, true_demand
        pred_demand = np.apply_along_axis((lambda x:log_means_predict_demand(x, \
            labels1_mean, labels2_mean, labels3_mean, all_mean, \
            labels1_idx, labels2_idx, labels3_idx, mult_factor, plus_factor)), \
            1, df_test[labels].values)
        start_time = time.time()
        true_demand = df_test['Demanda_uni_equil'].values
        RMSLE = np.sqrt(MSE(np.log1p(pred_demand), np.log1p(true_demand)))
        print 'RMSLE=', RMSLE
        print 'RMSLE calculating time=', time.time()-start_time

    print 'predicting time=', time.time()-start_time

if __name__ == '__main__':
    # sample_test = int(sys.argv[2])==1
    config_path = sys.argv[1]
    vali = int(sys.argv[2])==1
    config_name_list = sys.argv[3::]
    start_time = time.time()
    print 'reading data'

    df_train = pandas.read_csv('data/train_join.csv')
    df_test = pandas.read_csv('data/test_join.csv')
    df_train['log_Demanda_uni_equil'] = df_train['Demanda_uni_equil'].apply(lambda x:np.log1p(x))

    print 'reading time=', time.time()-start_time
    if vali:
        time_quantile = 7 # train: week 3~7 test: week 8,9
        df_test = df_train[df_train['Semana']>time_quantile]
        df_train = df_train[df_train['Semana']<=time_quantile]

    for config_name in config_name_list:
        if not vali:
            group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali)
        else:
            group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali)
        print '\n\n------------------------------------------\n\n'
