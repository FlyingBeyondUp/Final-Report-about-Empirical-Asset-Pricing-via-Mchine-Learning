# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 00:05:21 2025

@author: zy01
"""
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.base import clone 
from multiprocessing.shared_memory import SharedMemory
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler,PowerTransformer,power_transform
import multiprocessing
import gc
import matplotlib.pyplot as plt

class Scaler:
    def __init__(self,n_cat=74):
        self.scaler=StandardScaler()
        self.n_cat=n_cat
    
    def fit_transform(self,X):
        self.scaler.fit(X[:,:X.shape[1]-self.n_cat])
        X_real_valued=self.scaler.transform(X[:,:X.shape[1]-self.n_cat]).astype(np.float32)
        return np.concatenate((X_real_valued,X[:,X.shape[1]-self.n_cat:]),axis=1).astype(np.float32)
    
    def transform(self,X):
        X_real_valued=self.scaler.transform(X[:,:X.shape[1]-self.n_cat]).astype(np.float32)
        return np.concatenate((X_real_valued,X[:,X.shape[1]-self.n_cat:]),axis=1).astype(np.float32)
    
# def get_scaled(X,n_cat=74,scaler=None):
#     if scaler is None:
#         scaler=StandardScaler()
#         scaler.fit(X[:,:X.shape[1]-n_cat])
#     X_real_valued=scaler.transform(X[:,:X.shape[1]-n_cat])
#     # X_scaled_real_valued=scaler.transform(X[:,:X.shape[1]-74])
#     # X_real_valued=pd.DataFrame(X[:,:X.shape[1]-74])
#     # columns=X_real_valued.columns[abs(X_real_valued.skew().values)>1]
#     # X_real_valued[columns]=power_transform(X_real_valued[columns].values)
#     # X_real_valued=scaler.fit_transform(X_real_valued.values)
#     return np.concatenate((X_real_valued,X[:,X.shape[1]-n_cat:]),axis=1)


def getR2(Y_pred:np.array,Y_test:np.array)->float:
    return 1-np.linalg.norm(Y_pred-Y_test,2)**2/np.linalg.norm(Y_test,2)**2


def get_subgroup_R2(Y_pred,Y_test,market_value,start,end,num_sum_year,n=int(1000*12)):
    Y_pred_large,Y_pred_small,Y_test_large,Y_test_small=None,None,None,None
    for i in range(start,end):
        ids=np.argsort(market_value[num_sum_year[i]:num_sum_year[i+1]])
        if Y_pred_large is None:
            Y_pred_large=Y_pred[ids[len(ids)-n:]]
            Y_pred_small=Y_pred[ids[:n]]
            Y_test_large=Y_test[ids[len(ids)-n:]]
            Y_test_small=Y_test[ids[:n]]
        else:
            Y_pred_large=np.concatenate((Y_pred_large,Y_pred[ids[len(ids)-n:]]))
            Y_pred_small=np.concatenate((Y_pred_small,Y_pred[ids[:n]]))
            Y_test_large=np.concatenate((Y_test_large,Y_test[ids[len(ids)-n:]]))
            Y_test_small=np.concatenate((Y_test_small,Y_test[ids[:n]]))
    return getR2(Y_pred_small,Y_test_small),getR2(Y_pred_large,Y_test_large)


def DM_test(Y_pred1,Y_pred2,Y_test,start,end,num_per_year,num_sum_year):
    list_d12=[]
    for i in range(start,end):
        start_i,end_i=num_sum_year[i]-num_sum_year[start],num_sum_year[i+1]-num_sum_year[start]
        d12=(np.linalg.norm(Y_pred1[start_i:end_i]-Y_test[start_i:end_i],2)**2-np.linalg.norm(Y_pred2[start_i:end_i]-Y_test[start_i:end_i],2)**2)/num_per_year[i]
        list_d12.append(d12)
    return np.mean(list_d12)/np.std(list_d12)



def recursive_evaluate_scaled(Model,args,Z,Y,num_sum_year,start,end,complexity=False,list_columns_to_mask=None,list_maksed_R2_reduction=[]):
    Y_pred=None
    list_complexity=[]
    for i in range(start,end):
        print(i)
        model=Model(args[i-start])
        
        scaler=Scaler()
        model.fit(scaler.fit_transform(Z[:num_sum_year[i],:]),Y[:num_sum_year[i]])
        
        if complexity:
            list_complexity.append(model.complexity)
            
        Y_pred_i=model.predict(scaler.transform(Z[num_sum_year[i]:num_sum_year[i+1],:]))
        
        if list_columns_to_mask is not None:
            list_R2_reduction=[]
            R2=getR2(model.predict(scaler.transform(Z[num_sum_year[i-1]:num_sum_year[i],:])),Y[num_sum_year[i-1]:num_sum_year[i]])
            for columns_to_mask in list_columns_to_mask:
                Z_i_masked=scaler.transform(Z[num_sum_year[i-1]:num_sum_year[i],:])
                Z_i_masked[:,columns_to_mask]=0
                Y_pred_i_masked=model.predict(Z_i_masked)
                list_R2_reduction.append(R2-getR2(Y_pred_i_masked,Y[num_sum_year[i-1]:num_sum_year[i]]))
            list_maksed_R2_reduction.append(list_R2_reduction)
            
        if Y_pred is None:
            Y_pred=Y_pred_i
        else:
            Y_pred=np.concatenate((Y_pred,Y_pred_i))    
    return getR2(Y_pred,Y[num_sum_year[start]:num_sum_year[end]]),Y_pred.astype(np.float32),list_complexity



def recursive_evaluate(model,Z,Y,num_sum_year,start,end,complexity=False,num_processes=1):
    if num_processes==1:
        Y_pred=None
        list_complexity=[]
        for i in range(start,end):
            print(i)
            model.fit(Z[:num_sum_year[i],:],Y[:num_sum_year[i]])
            if complexity:
                list_complexity.append(model.complexity)
            Y_pred_i=model.predict(Z[num_sum_year[i]:num_sum_year[i+1],:])
            if Y_pred is None:
                Y_pred=Y_pred_i
            else:
                Y_pred=np.concatenate((Y_pred,Y_pred_i))
    elif num_processes>1:
        Z_shm=SharedMemory(create=True,size=Z.nbytes)
        Y_shm=SharedMemory(create=True,size=Y.nbytes)
        npy_shm=SharedMemory(create=True,size=num_sum_year.nbytes)
        
        Z_shared=np.ndarray(Z.shape,dtype=Z.dtype,buffer=Z_shm.buf)
        Y_shared=np.ndarray(Y.shape,dtype=Y.dtype,buffer=Y_shm.buf)
        npy_shared=np.ndarray(num_sum_year.shape,dtype=num_sum_year.dtype,buffer=npy_shm.buf)
        
        Z_shared[:,:]=Z[:,:]
        Y_shared[:]=Y[:]
        npy_shared[:]=num_sum_year[:]
        
        pool=Pool(processes=num_processes,initializer=init_worker,initargs=(Z_shm.name, Z.shape, Z.dtype,
                               Y_shm.name, Y.shape, Y.dtype,npy_shm.name,num_sum_year.shape,num_sum_year.dtype))
        args=[(clone(model),i) for i in range(start,end)]
        list_Y_pred=pool.map(single_evaluate,args)
        pool.close()
        pool.join()
        
        Z_shm.close()
        Y_shm.close()
        npy_shm.close()

        Z_shm.unlink()
        Y_shm.unlink()
        npy_shm.unlink()
        Y_pred=np.concatenate(list_Y_pred)
    return getR2(Y_pred,Y[num_sum_year[start]:num_sum_year[end]]),Y_pred.astype(np.float32),list_complexity

def single_evaluate(Args):
    global Z_shared, Y_shared, npy_shared  # Access global shared arrays
    model, i = Args
    model.fit(Z_shared[:np.sum(npy_shared[:i]), :], Y_shared[:np.sum(npy_shared[:i])])
    gc.collect()  # add garbage collection
    return model.predict(Z_shared[np.sum(npy_shared[:i]):np.sum(npy_shared[:i + 1]), :])

def init_worker(Z_shm_name, Z_shape, Z_dtype, Y_shm_name, Y_shape, Y_dtype,npy_shm_name,npy_shape,npy_dtype):
    """Initialize worker processes by attaching to shared memory."""
    global Z_shared, Y_shared,npy_shared  # Define Z_shared and Y_shared as global for the worker

    # Attach to shared memory
    Z_shm = SharedMemory(name=Z_shm_name)
    Y_shm = SharedMemory(name=Y_shm_name)
    npy_shm=SharedMemory(name=npy_shm_name)

    # Create NumPy arrays backed by shared memory
    Z_shared = np.ndarray(Z_shape, dtype=Z_dtype, buffer=Z_shm.buf)
    Y_shared = np.ndarray(Y_shape, dtype=Y_dtype, buffer=Y_shm.buf)
    npy_shared=np.ndarray(npy_shape,dtype=npy_dtype,buffer=npy_shm.buf)

    # Close the local shared memory object (but keep the shared memory block alive)
    Z_shm.close()
    Y_shm.close()
    npy_shm.close()

def sort_lists_by_descending_order(A, B):
    """
    Sorts list A in descending order and sorts list B according to the sorted order of A,
    maintaining the original relationship between elements in A and B.

    Args:
        A (list): The list to be sorted in descending order.
        B (list): The list to be sorted according to the sorted order of A.  Must be the same length as A.

    Returns:
        tuple: A tuple containing the sorted list A and the sorted list B.

    Raises:
        ValueError: If the lists A and B have different lengths.
    """

    if len(A) != len(B):
        raise ValueError("Lists A and B must have the same length.")

    # 1. Create a list of tuples, where each tuple contains an element from A and its corresponding element from B.
    combined_list = list(zip(A, B))

    # 2. Sort the combined list in descending order based on the elements from A.
    #    Use a lambda function as the key to specify that the sorting should be based on the first element of each tuple (element from A).
    combined_list.sort(key=lambda x: x[0], reverse=True)

    # 3. Unzip the sorted combined list back into separate lists A and B.
    sorted_A, sorted_B = zip(*combined_list)

    # 4. Convert the tuples back to lists
    sorted_A = list(sorted_A)
    sorted_B = list(sorted_B)

    return sorted_A, sorted_B


def plot_horizontal_bar(strings, floats, title="", xlabel="Variable Importance", ylabel="features"):
    """
    Plots a horizontal bar chart with strings on the y-axis and float numbers on the x-axis.

    Args:
        strings (list of str): List of strings for the y-axis labels.
        floats (list of float): List of float numbers for the x-axis values.  Must be the same length as strings.
        title (str, optional): Title of the plot. Defaults to "Horizontal Bar Plot".
        xlabel (str, optional): Label for the x-axis. Defaults to "Values".
        ylabel (str, optional): Label for the y-axis. Defaults to "Categories".
    """

    if len(strings) != len(floats):
        raise ValueError("The lists of strings and floats must have the same length.")

    # 1. Create the plot
    fig, ax = plt.subplots()

    # 2. Create the y-axis positions (reversed for better readability)
    y_pos = np.arange(len(strings))

    # 3. Plot the bars
    ax.barh(y_pos, floats, align='center')  # barh for horizontal bars

    # 4. Set the y-axis labels (strings)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strings)

    # 5. Invert the y-axis for better readability (optional, but common)
    ax.invert_yaxis()  # labels read top-to-bottom

    # 6. Set the axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)

    # 7. Add grid lines (optional)
    ax.grid(True, axis='x', linestyle='--') # Add grid lines along the x axis
    plt.savefig(title)

    # 8. Show the plot
    plt.show()