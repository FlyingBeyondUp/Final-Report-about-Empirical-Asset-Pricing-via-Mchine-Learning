# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:54:42 2025

@author: LZ166
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from utils import recursive_evaluate,getR2,get_subgroup_R2,recursive_evaluate_scaled
from models import myPCR,TorchRegressor,NN1,NN2,NN3,NN4,NN5,myElasticNet,myPLS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor


Z=np.load('features_scaled.npy')
num_per_year=np.load('num_per_year_s.npy')
Y=np.load('Y_s.npy')
num_sum_year=np.array([np.sum(num_per_year[:i+1]) for i in range(0,len(num_per_year))])
start,end=10,15

list_params_xgb=[(100,0.01),(200,0.01),(400,0.01),(100,0.1),(200,0.1),(400,0.1)]
list_R2_xgb=[]
for (n_estimators,lr) in list_params_xgb:
    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',  # or another regression objective
        n_estimators=n_estimators,  # Number of boosting rounds
        learning_rate=lr,
        max_depth=2,
        tree_method='hist',  # or 'approx', or 'exact' - Use CPU-based histogram algorithm or other CPU methods
        #device='cpu', # Remove or comment out this line; CPU is the default if tree_method is not 'gpu_hist'
        random_state=42
    )
    R2,_=recursive_evaluate(model_xgb, Z, Y, num_sum_year, start=10, end=15,num_processes=1)
    list_R2_xgb.append(R2)
np.save('R2_xgb.npy',list_R2_xgb)
# R2_large,R2_small=get_subgroup_R2(Y_pred,Y[num_sum_year[9]:num_sum_year[14]])

list_max_features=[5,10,20,30]
list_R2_RF=[]
for max_features in list_max_features:
    model_RF=RandomForestRegressor(max_depth=6,max_features=max_features,n_estimators=300,n_jobs=10)
    R2_RF,_=recursive_evaluate(model_RF, Z, Y, num_sum_year, start, end)
    list_R2_RF.append(R2_RF)
np.save('RF_R2.npy',list_R2_RF)

list_epsilon=[1.35,10,50,100,1000]
list_R2_Huber=[]
for epsilon in list_epsilon:
    model_Huber=HuberRegressor(epsilon=epsilon,alpha=0)
    R2_Huber,_=recursive_evaluate(model_Huber, Z, Y, num_sum_year, start, end)
    list_R2_Huber.append(R2_Huber)
np.save('Huber_R2', list_R2_Huber)
model_ols=LinearRegression()
R2_ols,_=recursive_evaluate(model_ols, Z, Y, num_sum_year, start, end)
# R2,Y_pred=recursive_evaluate(model, Z, Y, num_per_year, start=10, end=15,num_processes=1)


alpha = 1e-1  # Regularization strength (lambda)
l1_ratio = 0.5  # Balance between L1 (Lasso) and L2 (Ridge) regularization
list_params_EN=[(1e-1,0.5),(1e-2,0.5),(1e-3,0.5),(1e-4,0.5)]
list_R2_EN_validation=[]
for (alpha,l1_ratio) in list_params_EN:
    model_EN = myElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model_EN = SGDRegressor(
        loss='huber',        # Huber‐loss
        penalty='elasticnet',# L1+L2 penalty
        alpha=alpha,          # overall regularization strength
        l1_ratio=0.5,        # fraction of L1 in the penalty
        epsilon=1.35,        # Huber transition point
        max_iter=10,
        warm_start=True,
        learning_rate='optimal',
        tol=1e-2,
        random_state=0
    )
    R2,_=recursive_evaluate_scaled(model_EN, Z, Y, num_sum_year, start, end)
    list_R2_EN_validation.append(R2)
np.save('EN_R2_val.npy',list_R2_EN_validation)





list_K=[5,10,15,20]
list_R2_PLS_validation=[]
for K in list_K:
    model_pls = PLSRegression(n_components=K)
    R2_pls,_=recursive_evaluate(model_pls, Z, Y, num_sum_year, start, end)
    list_R2_PLS_validation.append(R2_pls)
np.save('PLS_R2_val.npy',list_R2_PLS_validation)
    

list_K=[10,30,50]
list_R2_PCR_validation=[]
for K in list_K:
    model_pcr = myPCR(n_components=K)
    R2_pcr,_=recursive_evaluate(model_pcr, Z, Y, num_sum_year, start, end)
    list_R2_PCR_validation.append(R2_pcr)
np.save('PCR_R2_val.npy',list_R2_PCR_validation)


input_size=Z.shape[1]
list_NNs=[NN1(input_size),NN2(input_size),NN3(input_size),NN4(input_size),NN5(input_size)]
list_params_NN=[(1e-2,1e-3),(1e-2,1e-4),(1e-2,1e-5),(1e-3,1e-3),(1e-3,1e-4),(1e-3,1e-5)]
list_R2_NN=[]
for lr,l1_ratio in list_params_NN:
    model_NN=TorchRegressor(NN1(input_size),lr=lr,l1_ratio=l1_ratio,epochs=200)
    R2_NN,Y_pred_NN=recursive_evaluate(model_NN, Z, Y, num_sum_year, start, end)
    list_R2_NN.append(R2_NN)
np.save('R2_NN.npy',list_R2_NN)
