import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform,StandardScaler
import gc





# The macro predictors include:
# dividend-price ratio (dp)= d12/price, earnings-price ratio (ep)= e12/price, book-to-market ratio (bm), 
# net equity expansion (ntis), Treasury-bill rate (tbl), term spread (tms)= lty - tbl, 
# default spread (dfy)= BAA - AAA, and stock variance (svar)
def getZ(df,df_macro):
    Z=None
    list_num_per_month=[]
    for date in df_macro.yyyymm:
        macro_data=df_macro[df_macro.yyyymm==date]
        dp=macro_data.D12/macro_data.Index
        ep=macro_data.E12/macro_data.Index
        tms=macro_data.lty-macro_data.tbl
        dfy=macro_data.BAA-macro_data.AAA
        x=np.array([1,dp.item(),ep.item(),macro_data.bm.item(),macro_data.ntis.item(),macro_data.tbl.item(),tms.item(),dfy.item(),macro_data.svar.item()],dtype=np.float32)
        
        df_at_date=df[df.DATE//1e2==date]
        for column in df.columns[2:-1]:
            if df_at_date[column].isnull().all():
                df_at_date.loc[:,column]=df_at_date[column].fillna(0) # if all of the data in the column is nan, then discard it at this stage.
            else:
                df_at_date.loc[:,column]=df_at_date[column].fillna(df_at_date[column].median())
        C=np.array(df_at_date[df_at_date.columns[2:-1]],dtype=np.float32)
        Z_at_date=np.kron(C,x.reshape(1,9))
        list_num_per_month.append(Z_at_date.shape[0])
        
        sic2_encode=np.zeros((C.shape[0],74),dtype=bool)  
        col_indices = df_at_date['sic2'].values.astype(int)  # Get the SIC2 values as column indices
        sic2_encode[np.arange(C.shape[0]), col_indices-1] = True
        
        Z_at_date=np.concatenate((Z_at_date,sic2_encode),axis=1)
        if Z is None:
            Z=Z_at_date
        else:
            Z=np.concatenate((Z,Z_at_date))
    num_per_month=np.array(list_num_per_month)
    num_per_year=num_per_month.reshape(int(num_per_month.shape[0]/12),12).dot(np.ones((12,1)))
    return Z,num_per_year.astype(int),num_per_month

if __name__ == '__main__':
    # df=pd.read_csv('datashare.csv')
    df=pd.read_csv('GKX_20201231.csv')
    # df['RET']=df_with_Y.RET
    # del df_with_Y
    gc.collect()
    date_stock=df.DATE
    start_year_index=date_stock[date_stock//1e4==1990].index[0]
    end_year_index=date_stock[date_stock//1e4==2015].index[0]
    df=df.iloc[start_year_index:end_year_index]
    
    df=df[df['sic2'].notna()]
    df=df[df.sic2<=74]
    # market_value=df.prc*df.SHROUT
    sic2s=df.sic2
    
    # df=df[market_value.notna()]
    Y=df.RET.astype(np.float32)
    
    del df['sic2'],df['RET'],df['prc'],df['SHROUT']
    df['sic2']=sic2s
    
    list_top_features=["mom1m","mom12m","chmom","indmom","maxret","mom36m",
                       "turn","std_turn","mvel1","dolvol","ill","zerotrade",
                       "baspread","retvol","idiovol","beta","betasq","ep",
                       "sp","agr","nincr"]
    column_indices = [df.columns.get_loc(col)-2 for col in list_top_features] # the first two features are id and year
    
    ols_3_ids=[df.columns.get_loc('mvel1')-2,df.columns.get_loc('bm')-2,df.columns.get_loc('mom1m')-2]

    
    # skews=df.skew()
    # maxs=df.max()
    # mins=df.min()
    
    # df[df.columns[2:-1]]=power_transform(df[df.columns[2:-1]].values)
    # scaler=StandardScaler()
    # df[df.columns[2:-1]]=scaler.fit_transform(df[df.columns[2:-1]].values)
    
    
    df_macro=pd.read_excel("PredictorData2023.xlsx")
    start_year_index=df_macro.yyyymm[df_macro.yyyymm//1e2==1990].index[0]
    df_macro=df_macro[start_year_index:]
    df_macro=df_macro.rename(columns={'b/m':'bm'})
    df_macro=df_macro[df_macro.yyyymm.values//1e2<=2014]
    
    Z,num_per_year,num_per_month=getZ(df,df_macro)
    
    list_Z_ols_3=[Z[:,9*ols_3_ids[0]].reshape(-1,1),Z[:,9*ols_3_ids[1]].reshape(-1,1),Z[:,9*ols_3_ids[2]].reshape(-1,1)]
    Z_ols_3_nonnan=np.concatenate(list_Z_ols_3,axis=1)
    
    del df,df_macro
    np.save('features_25y.npy', Z)
    np.save('num_per_year_25y.npy',num_per_year)
    np.save('num_per_month_25y.npy',num_per_month)
    np.save('Y_25y.npy',Y)
    # np.save('market_value_25y.npy',market_value)
    np.save('column_indices.npy',column_indices)
    np.save('Z_ols_3.npy',list_Z_ols_3)
    


