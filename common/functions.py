#!/usr/bin/env python
# encoding: utf-8

'''
Common unclassified functions
'''
from pylab import *
import numpy as np
import pandas as pd
from itertools import chain
from collections import deque
import sys
from datetime import datetime


def row(x):
    '''Given sequence x returns numpy array x as row-vector with shape (1,len(x))
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    if not isinstance(x,np.ndarray):
        x=np.array(x)
    assert len(x.shape)==1,'x should contain only one axis!'
    return array(x)[np.newaxis,:]

def vec(x):
    '''Given sequence x returns numpy array x as vector-column with shape (len(x),1)
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    if not isinstance(x,np.ndarray):
        x=np.array(x)
    assert len(x.shape)==1,'x should contain only one axis!'
    return x[:,np.newaxis]
    
    
def normalize(z):
    '''Feature normalization
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    return (z-min(z))/(max(z)-min(z))
    
    
    
def all_nums(X):
    '''
    Prints if vector/series/matrix/dataframe consits only of numbers or contains inf or nan
    
    Example:
    
    A=eye(3)
    A[1,1]=inf
    A[2,2]=nan    
    A=pd.DataFrame(A,columns=['a','b','c'])
    all_nums(A)  # will return False and print that matrix is not composed only of numbers, 
                 # because it has 1 inf for column 'b' and one nan for column 'c'.
    
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    def all_num_array(X, col_name=None):
            try:
                X = np.array(np.array(X),dtype=np.float64)
            except ValueError:
                error = 'not numeric type'
            if all(np.isfinite(X)):
                error = ''
            else:
                if ~any(np.isinf(X)) and any(np.isnan(X)):
                    error = 'has %i nan' % np.isnan(X).ravel().sum() 
                if any(np.isinf(X)) and ~any(np.isnan(X)):
                    error = 'has %i inf' % np.isinf(X).ravel().sum() 
                if any(np.isinf(X)) and any(np.isnan(X)):
                    error = 'has %i nan and %i inf' % (np.isnan(X).ravel().sum(),np.isinf(X).ravel().sum()) 
            if error=='':  # all numbers
                return True 
            else:   # not all numbers
                if col_name is None:
                    print(error)
                else:
                    print('%s: %s'%(col_name,error))
                return False

    if isinstance(X,np.ndarray):

        if len(X.shape)==1:
            return all_num_array(X)

        elif len(X.shape)==2:
            res = np.zeros(X.shape[1],dtype=bool)
            for col in range(X.shape[1]):
                res[col] = all_num_array(X[:,col],'column%d'%col)
            if all(res):
                print('SUCCESS, all columns have numbers.\n')
            else:
                print('FAIL, not all columns have numbers.')
        else:
            raise ValueError('Number of dimensions of X is more than 2!')

    elif isinstance(X,(list,tuple,pd.Series)):
        return all_num_array(X)

    elif isinstance(X,(pd.DataFrame)):
        res = np.zeros(len(X.columns),dtype=bool)
        for i,col_name in enumerate(X.columns):
            res[i] = all_num_array(X[col_name].values, col_name)
        if all(res):
            print('SUCCESS, all columns have numbers.\n')
        else:
            print('FAIL, not all columns have numbers')
    else:
        raise ValueError('Invalid type!')
        
        
     

     
        
     