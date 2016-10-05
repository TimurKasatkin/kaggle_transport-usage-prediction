#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from common.functions import vec, row
from pylab import *




def get_one_hot_encoding(x):
    '''
    INPUT:
        x: pandas series

    OUTPUT:
        pandas dataframe df with one-hot encoded values of x

    COMMENTS:
        df preserves index of x.

    EXAMPLE:
        A=pd.Series(['new','old','new','old','moderate','moderate'],index=[11,12,13,14,15,16],name='status')
    TRANSFORMS TO:
            status=moderate	status=new	status=old
        11	0	1	0
        12	0	0	1
        13	0	1	0
        14	0	0	1
        15	1	0	0
        16	1	0	0
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.        
    '''

    assert x.name is not None,'Series should contain name!'
    values = np.unique(x)
    values2nums = dict( zip(values,list(range(len(values)))) )
    nums2values = dict( zip(values2nums.values(),values2nums.keys()) )
    x = x.replace(values2nums)
    encoder = skl.preprocessing.OneHotEncoder(sparse=False)
    data = encoder.fit_transform(x.values[:,np.newaxis])
    columns = ['%s=%s' % (x.name,nums2values[value]) for value in encoder.active_features_]
    return pd.DataFrame(data, index=x.index, columns=columns, dtype=np.int8)



def get_probability_encoding(feature,output,fill_nans=True, missing_value=None):
    '''
    Input:
    feature - feature (pandas series)
    output - class output (pandas series)
    fill_nans - fill nan rows (undefined probabilities) with general class priors
    Output:
    pandas dataframe of probabilities of all classes given each value of feature

    example1:
        get_probability_encoding(f=[10,10,20,20,30,30,nan,nan,10,20,30],Y=[0,0,1,1,0,1,0,1,nan,nan,nan], fill_nans=False)
    returns:
        (array([[ 1. ,  0. ],
                [ 1. ,  0. ],
                [ 0. ,  1. ],
                [ 0. ,  1. ],
                [ 0.5,  0.5],
                [ 0.5,  0.5],
                [ nan,  nan],
                [ nan,  nan],
                [ 1. ,  0. ],
                [ 0. ,  1. ],
                [ 0.5,  0.5]]), array([ 0.,  1.]))

    example2:
        get_probability_encoding(f=['a','a','b','b','c','c','?','?','a','b','c'],Y=[0,0,1,1,0,1,0,1,nan,nan,nan], fill_nans=False, missing_value='?')
    returns:
        (array([[ 1. ,  0. ],
                [ 1. ,  0. ],
                [ 0. ,  1. ],
                [ 0. ,  1. ],
                [ 0.5,  0.5],
                [ 0.5,  0.5],
                [ nan,  nan],
                [ nan,  nan],
                [ 1. ,  0. ],
                [ 0. ,  1. ],
                [ 0.5,  0.5]]), array([ 0.,  1.]))
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    f = feature.values
    Y = output.values

    assert len(f)==len(Y), 'length on features and length of outputs should be the same.'
    assert all(feature.index==output.index),'feature.index should be equal to output.index'
    assert hasattr(feature,'name'),'feature should be pandas Series with name property.'


    if (missing_value is None) or (missing_value is nan):
        valid_f_sels = ~isnan(f)
    else:
        valid_f_sels = (f!=missing_value)
        
    feature_vals = unique(f[valid_f_sels])
    Y_vals,counts = np.unique(Y[~isnan(Y)], return_counts=True)  # Y_vals correspond to columns of output matrix X
    y_probs = counts/np.sum(counts)

    X = np.zeros( (len(Y),len(Y_vals)) )
    X[~valid_f_sels,:]=nan
    
    f_values_y_not_defined = set(f[isnan(Y)])-set(f[~isnan(Y)])  # a set of feature values that appear only when Y is nan
    for f_val in f_values_y_not_defined:
        X[f==f_val,:]=nan

    for feature_val in feature_vals:
        (Y_cond_vals,Y_cond_counts) = unique(Y[~isnan(Y) & (f==feature_val)], return_counts=True)
        Y_cond_probs = Y_cond_counts/sum(Y_cond_counts)
        for Y_cond_val,Y_cond_prob in zip(Y_cond_vals,Y_cond_probs):
            X[f==feature_val, Y_vals==Y_cond_val] = Y_cond_prob

    if fill_nans:
        for i in range(X.shape[1]):
            X[np.isnan(X[:,i]),i]=y_probs[i]

    return pd.DataFrame(X,index=feature.index, columns=['P(y=%s|%s)'%(y_val, feature.name) for y_val in Y_vals])


    

