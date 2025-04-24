# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 19:18:42 2025

@author: zy01
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

from utils import recursive_evaluate,getR2,get_subgroup_R2,recursive_evaluate_scaled,DM_test,Scaler
from models import myPCR,myENet,myRF,myGBRT
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from utils import plot_horizontal_bar,sort_lists_by_descending_order

# read the feature matrix Z and target vector y, which are given by data_processing.py
Z=np.load('features_25y.npy')
Z_ols_3=np.load('Z_ols_3.npy')
num_per_year=np.load('num_per_year_25y.npy')
Y=np.load('Y_25y.npy')
market_value=np.load('market_value_25y.npy')
column_indices=np.load('column_indices.npy')
num_sum_year=np.array([np.sum(num_per_year[:i+1]) for i in range(0,len(num_per_year))])
start,end=15,24

Z_ols_3=pd.DataFrame(Z_ols_3)
for column in Z_ols_3.columns:
    Z_ols_3[column]=Z_ols_3[column].fillna(Z_ols_3[column].median())
Z_ols_3=np.array(Z_ols_3,dtype=np.float32)



###############################################################################
# Regularized Linear Models:


# list_models1=[(model_PLS,False),(model_EN,True),(model_PCR,False)]
list_models1=[(PLSRegression,[3,3,3,3,7,6,4,4,4],False),(myPCR,[40,40,40,40,80,40,80,80,90,90],False),(myENet,[0.01 for i in range(start,end)],True)]
list_models1_complexities=[]
list_Y_pred,list_R2=[],[]

for Model,args,complexity in list_models1:
    R2,Y_pred,list_complexities=recursive_evaluate_scaled(Model,args, Z, Y, num_sum_year, start, end,complexity)
    list_models1_complexities.append(list_complexities)
    list_R2.append(R2)
    list_Y_pred.append(Y_pred)
list_small_large_R2=[get_subgroup_R2(Y_pred, Y[num_sum_year[start]:num_sum_year[end]], market_value, start, end, num_sum_year) for Y_pred in list_Y_pred]
list_R2_RL=[R2 for R2 in list_R2]

###############################################################################
# Linear Models:
# model_ols=LinearRegression()
# R2_ols,Y_pred_ols,_=recursive_evaluate(model_ols, Z, Y, num_sum_year, start, end,complexity=False)
R2_ols=np.load('Huber_R2.npy')[-1]

# model_ols_3=HuberRegressor(epsilon=3.5)
# R2_ols_3,Y_pred_ols_3,_=recursive_evaluate_scaled(model_ols_3, Z_ols_3, Y, num_sum_year, start, end)
model_ols_3=LinearRegression()
R2_ols_3,Y_pred_ols_3,_=recursive_evaluate(model_ols_3, Z_ols_3, Y, num_sum_year, start, end)
list_R2_ols=[R2_ols,R2_ols_3]
list_Y_pred_ols=[Y_pred_ols_3]

np.save('Y_pred_ols_3.npy',list_Y_pred_ols)
# list_Y_pred_ols=list(np.load('Y_pred_ols.npy'))

list_small_large_R2_ols=[get_subgroup_R2(Y_pred, Y[num_sum_year[start]:num_sum_year[end]], market_value, start, end, num_sum_year) for Y_pred in list_Y_pred_ols]




###############################################################################
# Tree Models
model_rf=myRF(max_depth=6,max_features=30)
model_xgb=myGBRT(max_depth=2, lr=0.01)
list_models2=[(model_xgb,True),(model_rf,True)]
list_models2_complexities=[]
list_Y_pred2,list_R2_2=[],[]

for model,complexity in list_models2:
    R2,Y_pred,list_complexities=recursive_evaluate(model, Z, Y, num_sum_year, start, end,complexity)
    list_models2_complexities.append(list_complexities)
    list_R2_2.append(R2)
    list_Y_pred2.append(Y_pred)
list_small_large_R2_2=[(get_subgroup_R2(Y_pred, Y[num_sum_year[start]:num_sum_year[end]], market_value, start, end, num_sum_year) for Y_pred in list_Y_pred2)]

###############################################################################
# summarize:
list_subgroup_R2=list_small_large_R2_ols+list_small_large_R2+list_small_large_R2_2
list_R2=list_R2_ols+list_R2_RL+list_R2_2
list_Y_pred_total=list_Y_pred_ols+list_Y_pred+list_Y_pred2

list_R2_total=[(list_R2[i+1],list_subgroup_R2[i][0],list_subgroup_R2[i][1]) for i in range(0,len(list_subgroup_R2))]
np.save('R2_bar.npy',list_R2_total)
R2_bar=np.load('R2_bar.npy')

list_DM_score=[]
for i in range(0,len(list_Y_pred_total)-1):
    list_DM_score_i=[]
    for j in range(i+1,len(list_Y_pred_total)):
        DM_score=DM_test(Y_pred1=list_Y_pred_total[i], Y_pred2=list_Y_pred_total[j], Y_test=Y[num_sum_year[start]:num_sum_year[end]], start=start, end=end, num_per_year=num_per_year, num_sum_year=num_sum_year)
        list_DM_score_i.append(DM_score)
    list_DM_score.append(list_DM_score_i)
import json
with open('DM_test.json', 'w') as f:  # 'w' for write text
    json.dump(list_DM_score, f)
np.save('model_complexities_ENet_GBRT.npy',list_models1_complexities[2]+list_models2_complexities[0])
models_complexities=np.load('model_complexities_ENet_GBRT.npy')


list_R2_total=np.load('R2_all.npy')
data=list_R2_total
bar_width = 0.2
r1 = np.arange(len(data))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Extract data for each bar group
group1 = [100*x[0] for x in data]
group2 = [100*x[2] for x in data]
group3 = [100*x[1] for x in data]

# Make the plot
plt.figure(figsize=(8, 5))
plt.bar(r1, group1, color='blue', width=bar_width, edgecolor='white', label='All')
plt.bar(r2, group2, color='green', width=bar_width, edgecolor='white', label='Top')
plt.bar(r3, group3, color='orange', width=bar_width, edgecolor='white', label='Bottom')

# Add labels, title, and legend
plt.xlabel('Algorithm')
plt.xticks([r + bar_width for r in range(len(data))], ['ols-3', 'PLS', 'PCR', 'ENet+H','GBRT','RF'])  # Example group labels
plt.ylabel(r'$R^2_{\mathrm{oos}}$')   
plt.legend()
plt.savefig('R2_bar.pdf',bbox_inches='tight')
    


##############################################################################
# feature importance study
# macro features:dp.item(),ep.item(),macro_data.bm.item(),macro_data.ntis.item(),macro_data.tbl.item(),tms.item(),dfy.item(),macro_data.svar.item()
# list_macro_features=['dp','ep','bm','ntis','tbl','tms','dfy','svar']
# list_macro_importance=[]
# GBRT=myGBRT(max_depth=2, lr=0.01)
# for i in range(1,9):   # iterate over all macro-economic feature
#     print(i)
#     columns_to_keep = np.ones(Z.shape[1], dtype=bool)  # Start with all columns True
#     column_delete_indices=[9*j+i for j in range(0,int((Z.shape[1]-74)/9))]
#     columns_to_keep[column_delete_indices] = False  # Set columns to remove to False
    
#     list_R2_RL_delete_=[]
#     for Model,args,_ in list_models1:
#         R2,_,_=recursive_evaluate_scaled(Model,args,Z[:, columns_to_keep], Y, num_sum_year, start, end)
#         list_R2_RL_delete_.append(R2)
#     list_feature_delete_=[list_R2_RL[i]-list_R2_RL_delete_[i] for i in range(0,len(list_R2_RL))]
#     R2,_,_=recursive_evaluate(GBRT, Z[:, columns_to_keep], Y, num_sum_year, start, end)
#     list_feature_delete_.append(R2-list_R2_2[0])
#     list_macro_importance.append(list_feature_delete_)


list_macro_features=['dp','ep','bm','ntis','tbl','tms','dfy','svar']
list_macro_importance=[]
GBRT=myGBRT(max_depth=2, lr=0.01)

list_columns_to_mask=[]
for i in range(1,9):   # iterate over all macro-economic feature
    column_mask_indices=[9*j+i for j in range(0,int((Z.shape[1]-74)/9))]
    list_columns_to_mask.append(column_mask_indices)
    
list_columns_to_mask=[]
for i in range(0,len(column_indices)):   # iterate over all macro-economic feature
    column_mask_indices=[9*column_indices[i]+j for j in range(0,9)]
    list_columns_to_mask.append(column_mask_indices)

list_models_masked_R2_reduction=[]
for Model,args,_ in list_models1:
    list_masked_R2_reduction=[]
    _,_,_=recursive_evaluate_scaled(Model,args, Z, Y, num_sum_year, start, end,list_columns_to_mask=list_columns_to_mask,list_maksed_R2_reduction=list_masked_R2_reduction)
    list_models_masked_R2_reduction.append(np.mean(np.array(list_masked_R2_reduction),axis=0))


scaler=Scaler()
model_PLS=PLSRegression(3)
model_PLS.fit(scaler.fit_transform(Z[:num_sum_year[start],:]),Y[:num_sum_year[start]])

importance_PLS=model_PLS.coef_
stock_features_importance_PLS=np.zeros(95)
for i in range(0,Z.shape[1]-74):
    stock_features_importance_PLS[int(i/9)]+=importance_PLS[0][i]
selected_features_PLS=stock_features_importance_PLS[column_indices]
selected_features_PLS=abs(selected_features_PLS)/np.sum(abs(selected_features_PLS))



scaler=Scaler()
model_EN=myENet(alpha=0.0001)
model_EN.fit(scaler.fit_transform(Z[:num_sum_year[start],:]),Y[:num_sum_year[start]])

importance_EN=model_EN.model.coef_
stock_features_importance_EN=np.zeros(95)
for i in range(0,Z.shape[1]-74):
    stock_features_importance_EN[int(i/9)]+=importance_EN[i]
selected_features_EN=stock_features_importance_EN[column_indices]
selected_features_EN=abs(selected_features_EN)/np.sum(abs(selected_features_EN))

importance_EN_macro=model_EN.model.coef_
macro_features_importance_EN=np.zeros(9)
for i in range(0,Z.shape[1]-74):
    macro_features_importance_EN[int(i%9)]+=importance_EN[i]
macro_features_importance_EN=abs(macro_features_importance_EN)/np.sum(abs( macro_features_importance_EN))
    



GBRT=myGBRT(max_depth=2, lr=0.1,n_estimators=400)
GBRT.fit(Z[:num_sum_year[start],:],Y[:num_sum_year[start]])
importance_gain = GBRT.model.get_booster().get_score(importance_type='total_gain')
keys = [int(key[1:]) for key in importance_gain.keys()]
values=[v for v in importance_gain.values()]
stock_features_importance=np.zeros(95)
for i,key in enumerate(keys):
    stock_features_importance[int(key/9)]+=values[i]
selected_features_GBRT=stock_features_importance[column_indices]
selected_features_GBRT=abs(selected_features_GBRT)/np.sum(abs(selected_features_GBRT))


list_top_features=["mom1m","mom12m","chmom","indmom","maxret","mom36m",
                    "turn","std_turn","mvel1","dolvol","ill","zerotrade",
                    "baspread","retvol","idiovol","beta","betasq","ep",
                    "sp","agr","nincr"]
sorted_VI,sorted_features=sort_lists_by_descending_order(selected_features_EN,list_top_features)
plot_horizontal_bar(sorted_features,sorted_VI,title='EN_VI.pdf')

