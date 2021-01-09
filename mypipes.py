import pandas as pd 
import numpy as np

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

class VarSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,var_names,drop_var=False):
        self.vars=var_names
        self.drop_var=drop_var
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,X):
        if self.drop_var:
            return X.drop(self.vars,1)
        else:
            return X[self.vars]



class string_clean(BaseEstimator, TransformerMixin):
    
    def __init__(self,replace_it='',replace_with=''):
        self.replace_it=replace_it
        self.replace_with=replace_with
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,X):
        for col in X.columns:
            X[col]=X[col].str.replace(self.replace_it,self.replace_with)
        return X



class convert_to_numeric(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,X):
        for col in X.columns:
            X[col]=pd.to_numeric(X[col],errors='coerce')
        return X


class get_dummies_Pipe(BaseEstimator, TransformerMixin):
    
    def __init__(self,freq_cutoff=0):
        self.freq_cutoff=freq_cutoff
        self.var_cat_dict={}
        
    def fit(self,x,y=None):
        data_cols=x.columns

        for col in data_cols:
            k=x[col].value_counts()
            if (k<=self.freq_cutoff).sum()==0:
                cats=k.index[k>self.freq_cutoff][:-1]

            else:
                cats=k.index[k>self.freq_cutoff]

            self.var_cat_dict[col]=cats
        return self
            
    def transform(self,x,y=None):
        dummy_data=x.copy()
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name=col+'_'+cat
                dummy_data[name]=(dummy_data[col]==cat).astype(int)
            del dummy_data[col]
        return dummy_data

        


class custom_fico(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,X):
        k=X['FICO.Range'].str.split("-",expand=True).astype(float)
        X['fico']=0.5*(k[0]+k[1])
        del X['FICO.Range']
        return X