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

def log_means_predict_demand(x, labels1_mean, labels2_mean, labels3_mean, all_mean, \
        labels1_idx, labels2_idx, labels3_idx, mult_factor, plus_factor):
    x_l1 = tuple(x[labels1_idx])
    x_l2 = tuple(x[labels2_idx])
    x_l3 = tuple(x[labels3_idx])

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

def log_means_predict_demand(x, labels1_mean, labels2_mean, labels3_mean, all_mean, \
        labels1_idx, labels2_idx, labels3_idx, mult_factor1, plus_factor1, \
        mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4, plus_factor4):
    x_l1 = tuple(x[labels1_idx])
    x_l2 = tuple(x[labels2_idx])
    x_l3 = tuple(x[labels3_idx])

    pred = 0
    if x_l1 in labels1_mean.index:
        pred = labels1_mean[x_l1]*mult_factor1+plus_factor1
    elif x_l2 in labels2_mean.index:
        pred = labels2_mean[x_l2]*mult_factor2+plus_factor2
    elif x_l3 in labels3_mean.index:
        pred = labels3_mean[x_l3]*mult_factor3+plus_factor3
    else:
        pred = all_mean*mult_factor4+plus_factor4

    return pred

## https://www.kaggle.com/apapiu/grupo-bimbo-inventory-demand/log-means/code
def group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels1 = config.get(config_name, 'labels1').split(',')
    labels2 = config.get(config_name, 'labels2').split(',')
    labels3 = config.get(config_name, 'labels3').split(',')
    labels = list(set(labels1 + labels2 + labels3))
    print config_name, ';', labels1, ';', labels2, ';', labels3, ';', labels
    labels1_idx = [labels.index(i) for i in labels1]
    labels2_idx = [labels.index(i) for i in labels2]
    labels3_idx = [labels.index(i) for i in labels3]
    # print labels1_idx, ';', labels2_idx, ';', labels3_idx

    mult_factor1 = mult_factor2 = mult_factor3 = mult_factor4 = 1
    plus_factor1 = plus_factor2 = plus_factor3 = plus_factor4 = 0
    if 'level' in config_name:
        mult_factor1, mult_factor2, mult_factor3, mult_factor4 = \
                [config.getfloat(config_name, 'mult_factor'+str(i)) for i in range(1,5)]
                # config.getfloat(config_name, 'mult_factor1'), \
                # config.getfloat(config_name, 'mult_factor2'), \
                # config.getfloat(config_name, 'mult_factor3'), \
                # config.getfloat(config_name, 'mult_factor4')
        plus_factor1, plus_factor2, plus_factor3, plus_factor4 = \
                [config.getfloat(config_name, 'plus_factor'+str(i)) for i in range(1,5)]
                # config.getfloat(config_name, 'plus_factor1'), \
                # config.getfloat(config_name, 'plus_factor2'), \
                # config.getfloat(config_name, 'plus_factor3'), \
                # config.getfloat(config_name, 'plus_factor4')
    else:
        mult_factor1 = mult_factor2 = mult_factor3 = mult_factor4 = \
                config.getfloat(config_name, 'mult_factor')
        plus_factor1 = plus_factor2 = plus_factor3 = plus_factor4 = \
                config.getfloat(config_name, 'plus_factor')
    print 'mult_factor1=', mult_factor1, ' plus_factor1=', plus_factor1
    print 'mult_factor2=', mult_factor2, ' plus_factor2=', plus_factor2
    print 'mult_factor3=', mult_factor3, ' plus_factor3=', plus_factor3
    print 'mult_factor4=', mult_factor4, ' plus_factor4=', plus_factor4

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
            labels1_idx, labels2_idx, labels3_idx, mult_factor1,plus_factor1, \
            mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4,plus_factor4)), \
            1, df_test[labels].values)
        df_test.to_csv('output/'+'_'.join(labels)+'_'+config_name+'_group_log_mean_'+ \
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['id','Demanda_uni_equil'], index=False)
    else:
        # global pred_demand, true_demand
        pred_demand = np.apply_along_axis((lambda x:log_means_predict_demand(x, \
            labels1_mean, labels2_mean, labels3_mean, all_mean, \
            labels1_idx, labels2_idx, labels3_idx, mult_factor1,plus_factor1, \
            mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4,plus_factor4)), \
            1, df_test[labels].values)
        true_demand = df_test['Demanda_uni_equil'].values
        RMSLE = np.sqrt(MSE(np.log1p(pred_demand), np.log1p(true_demand)))
        print 'RMSLE=', RMSLE

    print 'predicting time=', time.time()-start_time

if __name__ == '__main__':
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
        df_test = df_train[(df_train['Semana']>time_quantile) & \
            (df_train['Cliente_ID'].isin(pandas.unique(df_test['Cliente_ID'])))]
        df_train = df_train[df_train['Semana']<=time_quantile]
    # else: # test now
    #     df_train = df_train[(df_train['Cliente_ID'].isin(pandas.unique(df_test['Cliente_ID'])))]

    for config_name in config_name_list:
        if not vali:
            group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali)
        else:
            group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali)
        print '\n\n------------------------------------------\n\n'
