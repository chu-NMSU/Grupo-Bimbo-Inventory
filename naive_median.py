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

def log_means_pred_demand_func(x, labels1_mean, labels2_mean, labels3_mean, all_mean, \
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

def log_means_pred_demand_func(x, labels1_mean, labels2_mean, labels3_mean, all_mean, \
        labels1_idx, labels2_idx, labels3_idx, mult_factor1, plus_factor1, \
        mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4, plus_factor4):
    x_l1 = tuple(x[labels1_idx])
    x_l2 = tuple(x[labels2_idx])
    x_l3 = tuple(x[labels3_idx])

    pred = 0
    if x_l1 in labels1_mean.index:
        pred = np.expm1(labels1_mean[x_l1])*mult_factor1 + plus_factor1
    elif x_l2 in labels2_mean.index:
        pred = np.expm1(labels2_mean[x_l2])*mult_factor2 + plus_factor2
    elif x_l3 in labels3_mean.index:
        pred = np.expm1(labels3_mean[x_l3])*mult_factor3 + plus_factor3
    else:
        pred = np.expm1(all_mean)*mult_factor4 + plus_factor4

    return pred

def log_means_pred_demand_func(x, labels1_mean, labels2_mean, labels3_mean, all_mean, \
        labels1_idx, labels2_idx, labels3_idx, mult_factor11,mult_factor12, mult_factor1, \
        plus_factor1, mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4, plus_factor4):
    x_l1 = tuple(x[labels1_idx])
    x_l2 = tuple(x[labels2_idx])
    x_l3 = tuple(x[labels3_idx])

    pred = 0
    if mult_factor11!=-1 and mult_factor12!=-1 and \
            x_l1 in labels1_mean.index and x_l2 in labels2_mean.index:
        pred = (np.expm1(labels1_mean[x_l1])*mult_factor11 + \
                np.expm1(labels2_mean[x_l2])*mult_factor12)*mult_factor1+plus_factor1
    elif x_l1 in labels1_mean.index:
        pred = np.expm1(labels1_mean[x_l1])*mult_factor1 + plus_factor1
    elif x_l2 in labels2_mean.index:
        pred = np.expm1(labels2_mean[x_l2])*mult_factor2 + plus_factor2
    elif x_l3 in labels3_mean.index:
        pred = np.expm1(labels3_mean[x_l3])*mult_factor3 + plus_factor3
    else:
        pred = np.expm1(all_mean)*mult_factor4 + plus_factor4

    return pred

## https://www.kaggle.com/apapiu/grupo-bimbo-inventory-demand/log-means/code
def group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    labels1 = config.get(config_name, 'labels1').split(',')
    labels2 = config.get(config_name, 'labels2').split(',')
    labels3 = config.get(config_name, 'labels3').split(',')
    print config_name, ':', labels1, ';', labels2, ';', labels3

    mult_factor1 = mult_factor2 = mult_factor3 = mult_factor4 = 1
    plus_factor1 = plus_factor2 = plus_factor3 = plus_factor4 = 0
    if 'level' in config_name:
        mult_factor1, mult_factor2, mult_factor3, mult_factor4 = \
                [config.getfloat(config_name, 'mult_factor'+str(i)) for i in range(1,5)]
        plus_factor1, plus_factor2, plus_factor3, plus_factor4 = \
                [config.getfloat(config_name, 'plus_factor'+str(i)) for i in range(1,5)]
    else:
        mult_factor1 = mult_factor2 = mult_factor3 = mult_factor4 = \
                config.getfloat(config_name, 'mult_factor')
        plus_factor1 = plus_factor2 = plus_factor3 = plus_factor4 = \
                config.getfloat(config_name, 'plus_factor')
    mult_factor11 = mult_factor12 = -1.0
    if 'comb' in config_name:
        mult_factor11, mult_factor12 = \
                [config.getfloat(config_name, 'mult_factor'+str(i)) for i in [11,12]]

    print 'mult_factor11=', mult_factor11, ' mult_factor12=', mult_factor12
    print 'mult_factor1=', mult_factor1, ' plus_factor1=', plus_factor1
    print 'mult_factor2=', mult_factor2, ' plus_factor2=', plus_factor2
    print 'mult_factor3=', mult_factor3, ' plus_factor3=', plus_factor3
    print 'mult_factor4=', mult_factor4, ' plus_factor4=', plus_factor4

    predict(df_train, df_test, vali, labels1, labels2, labels3, mult_factor11, mult_factor12, \
            mult_factor1,plus_factor1, mult_factor2,plus_factor2, mult_factor3,plus_factor3,\
            mult_factor4,plus_factor4)

def predict(df_train, df_test, vali, labels1, labels2, labels3, mult_factor11, mult_factor12, \
            mult_factor1,plus_factor1, mult_factor2,plus_factor2, mult_factor3,plus_factor3,\
            mult_factor4,plus_factor4):
    start_time = time.time()
    labels = list(set(labels1 + labels2 + labels3))
    labels1_idx = [labels.index(i) for i in labels1]
    labels2_idx = [labels.index(i) for i in labels2]
    labels3_idx = [labels.index(i) for i in labels3]
    # print labels1_idx, ';', labels2_idx, ';', labels3_idx

    all_mean = df_train['log_Demanda_uni_equil'].mean()
    labels1_mean = df_train.groupby(by=labels1)['log_Demanda_uni_equil'].mean()
    labels2_mean = df_train.groupby(by=labels2)['log_Demanda_uni_equil'].mean()
    labels3_mean = df_train.groupby(by=labels3)['log_Demanda_uni_equil'].mean()

    print 'mean calculating time=', time.time()-start_time

    start_time = time.time()
    if not vali:
        df_test['Demanda_uni_equil']=np.apply_along_axis((lambda x:log_means_pred_demand_func(x,\
            labels1_mean, labels2_mean, labels3_mean, all_mean, \
            labels1_idx, labels2_idx, labels3_idx, mult_factor11,mult_factor12, mult_factor1,plus_factor1, \
            mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4,plus_factor4)), \
            1, df_test[labels].values)
        df_test.to_csv('output/'+'_'.join(labels)+'_'+config_name+'_group_log_mean_'+ \
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['id','Demanda_uni_equil'], index=False)
    else:
        # global pred_demand, true_demand
        pred_demand = np.apply_along_axis((lambda x:log_means_pred_demand_func(x, \
            labels1_mean, labels2_mean, labels3_mean, all_mean, \
            labels1_idx, labels2_idx, labels3_idx, mult_factor11, mult_factor12, mult_factor1,plus_factor1, \
            mult_factor2,plus_factor2, mult_factor3,plus_factor3, mult_factor4,plus_factor4)), \
            1, df_test[labels].values)
        true_demand = df_test['Demanda_uni_equil'].values
        RMSLE = np.sqrt(MSE(np.log1p(pred_demand), np.log1p(true_demand)))
        print 'RMSLE=', RMSLE

    print 'predicting time=', time.time()-start_time

def log_means_pred_demand_manual_func(x, P_mean, C_mean, PA_mean, PR_mean, PCA_mean, all_mean):
    x_l1 = (8,5,2)# tuple(x['Producto_ID', 'Cliente_ID', 'Agencia_ID'])
    x_l2 = (8,4)# tuple(x['Producto_ID', 'Ruta_SAK'])
    x_l3 = (8,2)# tuple(x['Producto_ID', 'Agencia_ID'])
    x_l4 = (5)# tuple(x['Cliente_ID'])
    x_l5 = (8)# tuple(x['Producto_ID'])

    pred = 0
    if x_l1 in PCA_mean.index and x_l2 in PR_mean.index:
        pred = (np.expm1(PCA_mean[x_l1])*0.71719 + \
                np.expm1(PR_mean[x_l2])*0.1825)*1.0 + 0.127
    elif x_l2 in PR_mean.index:
        pred = np.expm1(PR_mean[x_l2])*0.74 + 0.192
    elif x_l3 in C_mean.index:
        pred = np.expm1(C_mean[x_l3])*0.8219 + 0.85501
    elif x_l4 in PA_mean.index:
        pred = np.expm1(PA_mean[x_l4])*0.529 + 0.9501
    elif x_l5 in P_mean.index:
        pred = np.expm1(P_mean[x_l5])*0.489 + 0.9
    else:
        pred = np.expm1(all_mean)-0.909

    return pred

def group_log_means_predict_manual(df_train, df_test, vali):
    start_time = time.time()
    all_mean = df_train['log_Demanda_uni_equil'].mean()
    P_mean = df_train.groupby(by=['short_name'])['log_Demanda_uni_equil'].mean()
    C_mean = df_train.groupby(by=['Cliente_ID'])['log_Demanda_uni_equil'].mean()
    PA_mean = df_train.groupby(by=['short_name', 'Agencia_ID'])['log_Demanda_uni_equil'].mean()
    PR_mean = df_train.groupby(by=['short_name', 'Ruta_SAK'])['log_Demanda_uni_equil'].mean()
    PCA_mean = df_train.groupby(by=['short_name', 'Cliente_ID', 'Agencia_ID'])['log_Demanda_uni_equil'].mean()

    print 'mean calculating time=', time.time()-start_time

    start_time = time.time()
    if not vali:
        df_test['Demanda_uni_equil']=np.apply_along_axis((lambda x:log_means_pred_demand_manual_func(x,\
            P_mean, C_mean, PA_mean, PR_mean, PCA_mean, all_mean)), 1, df_test.values)
        df_test.to_csv('output/'+'manual_group_log_mean_'+ \
                str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
                columns=['id','Demanda_uni_equil'], index=False)
    else:
        # global pred_demand, true_demand
        pred_demand = np.apply_along_axis((lambda x:log_means_pred_demand_manual_func(x, \
            P_mean, C_mean, PA_mean, PR_mean, PCA_mean, all_mean)), 1, df_test[labels].values)
        true_demand = df_test['Demanda_uni_equil'].values
        RMSLE = np.sqrt(MSE(np.log1p(pred_demand), np.log1p(true_demand)))
        print 'RMSLE=', RMSLE

    print 'predicting time=', time.time()-start_time


def main(config_path, vali, config_name_list):
    start_time = time.time()
    print 'reading data'

    df_train = pandas.read_csv('data/train_join.csv')
    df_test = pandas.read_csv('data/test_join.csv')
    # df_train['log_Demanda_uni_equil'] = df_train['Demanda_uni_equil'].apply(lambda x:np.log1p(x))
    df_train['log_Demanda_uni_equil'] = 1.006999*df_train['Demanda_uni_equil'].apply(lambda x:np.log1p(x+0.011599))-0.011599

    print 'reading time=', time.time()-start_time
    if vali:
        time_quantile = 7 # train: week 3~7 test: week 8,9
        df_test = df_train[(df_train['Semana']>time_quantile) & \
            (df_train['Cliente_ID'].isin(pandas.unique(df_test['Cliente_ID'])))]
        df_train = df_train[df_train['Semana']<=time_quantile]
    # else: # test now
    #     df_train = df_train[(df_train['Cliente_ID'].isin(pandas.unique(df_test['Cliente_ID'])))]

    # for config_name in config_name_list:
    #     group_log_means_predict_pid(df_train, df_test, config_path, config_name, vali)
    #     print '\n\n------------------------------------------\n\n'
    group_log_means_predict_manual(df_train, df_test, vali)

if __name__ == '__main__':
    config_path = sys.argv[1]
    vali = int(sys.argv[2])==1
    config_name_list = sys.argv[3::]
    main(config_path, vali, config_name_list)

