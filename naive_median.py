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

# TODO refactor
def group_agg_cal(df_train, labels):
    start_time = time.time()
    print 'calculate group agg', labels
    N = len(labels)
    train_groups = df_train.groupby(by=labels)
    demand_group_agg = train_groups['Demanda_uni_equil'].median()
    demand_group_agg.to_csv('data/demand_group_agg_tmp.csv')
    return demand_group_agg

# TODO refactor
def predict_range(df_test, s_idx, labels):
    print 'predicting', s_idx, s_idx+10000 , '/', df_test.shape[0]
    SLE = 0
    for i in range(s_idx, min(s_idx+10000, df_test.shape[0])):
        test_labels = df_test.loc[df_test.index[i], labels].astype(str).values
        pred = 0
        for j in range(0, len(test_labels)):
            pred = count_dict[j+1][tuple(test_labels[0:j+1])]
            if pred!=0:
                break
        if pred==0:
            pred = 5
        if not vali:
            df_test.loc[df_test.index[i], 'Demanda_uni_equil'] = pred
        else:
            SLE += np.log(pred+1)-np.log(df_test.loc[df_test.index[i], 'Demanda_uni_equil']+1)
    if vali:
        # print 'RMSLE=', RMSLE
        return SLE
    else:
        return 0

# TODO refactor. parallel predicting
def group_mean_predict(df_train, df_test, vali, labels):
    start_time = time.time()

    global count_dict
    count_dict = dict()
    for i in range(len(labels)):
        count_dict[i+1] = group_agg_cal(df_train, labels[0:i+1])

    df_test['Demanda_uni_equil'] = 0
    SLEs = Parallel(n_jobs=48)(delayed (predict_range) (df_test, s, labels) \
            for s in np.arange(0, df_test.shape[0], 10000))
    RMSLE = np.sqrt(sum(SLEs)/df_test.shape[0])
    if vali:
        print 'RMSLE=', RMSLE

def predict_demand(x, product_agent_mean, product_mean, all_median):
    pid, aid, cid = x[['Producto_ID','Agencia_ID','Cliente_ID']]
    if (pid, aid, cid) in product_agent_mean.index:
        return product_agent_mean[(pid, aid, cid)]
    elif (pid, aid) in product_mean.index:
        return product_mean[(pid, aid)]
    else:
        return all_median

def kaggle_group_median_predict(df_train, df_test):
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
    df_test.to_csv('output/kaggle_group_mean_'+ \
            str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', \
            columns=['id','Demanda_uni_equil'], index=False)
    print 'predicting time=', time.time()-start_time

if __name__ == '__main__':
    # vali = int(sys.argv[1])==1
    # sample_test = int(sys.argv[2])==1
    start_time = time.time()
    print 'reading data'

    df_train = pandas.read_csv('data/train.csv')
    df_test = pandas.read_csv('data/test.csv')
    # if sample_test: # use 10% data to evaluate
    #     df_train = df_train[df_train['Cliente_ID']%100==9]
    df_train['log_Demanda_uni_equil'] = df_train['Demanda_uni_equil'].apply(lambda x:np.log(x+1))

    print 'reading time=', time.time()-start_time

    start_time = time.time()
    # if not vali and not sample_test:
    #     df_test = pandas.read_csv('data/test_join.csv')
    #     # df_sub = group_mean_predict(df_train, df_test, vali, labels)
    #     # df_sub.to_csv(columns=['id','Demanda_uni_equil'], path_or_buf='output/'+'-'.join(labels)+\
    #     #    '-'+str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.csv', index=False)
    # else:
    #     time_quantile = 7 # train: week 3~7 test: week 8,9
    #     df_test = df_train[df_train['Semana']>time_quantile]
    #     df_train = df_train[df_train['Semana']<=time_quantile]
    #     # df_sub = group_mean_predict(df_train, df_test, vali, labels)

    kaggle_group_median_predict(df_train, df_test)

    print 'total time=', time.time()-start_time
    print '\n\n------------------------------------------\n\n'
