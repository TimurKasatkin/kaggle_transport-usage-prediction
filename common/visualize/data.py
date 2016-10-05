import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from common.feature_transformations import get_lda_directions, get_lda_projections
from common.outliers import get_outlier_sels, get_outlier_sels_within_classes
from common.visualize.colors import COLORS
from pylab import *


    
    
def pca_report(X,Y=None):
    '''Find principal components for the data, plot the data in first 2 compoinents and show descriptive power of principal components.'''
    D=X.shape[1]
    pca =PCA()
    X_pca = pca.fit_transform(X_train)
    cum_explained_variance = cumsum(pca.explained_variance_ratio_)
    
    figure()
    if Y==None:
        scatter(X_pca[:,0], X_pca[:,1])
    else:
        scatter(X_pca[:,0], X_pca[:,1], c=[COLORS[y] for y in Y])
    xlabel('principal component 1')
    ylabel('principal component 2')
    title('First 2 components explain %.3f variance'%cum_explained_variance[1])
    
    figure()
    plot(range(1,D+1), cum_explained_variance)
    xticks( range(1,D+1) )
    xlabel('# components')
    ylabel('explained variance fraction')
    for threshold in [0.9, 0.95, 0.99]:
        ind = find(cum_explained_variance>threshold)[0]
        print('%.2f of variance is explained by %d components'% (threshold,ind+1))
        
        

def plot_corr(df,size=10,show_colorbar=True,show_grid=True):
    '''Function plots a graphical correlation matrix for earch pair of columns in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
        

    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }

    from matplotlib.colors import LinearSegmentedColormap
    blue_red_cmap = LinearSegmentedColormap('BlueRed1', cdict)


    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    plt.set_cmap(blue_red_cmap)
    m = ax.matshow(corr,interpolation='none', vmin=-1,vmax=1) #
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    if show_colorbar is True:
        plt.colorbar(m)
    if show_grid is True:
        plt.grid(color=[0.5,0.5,0.5], linestyle=':', linewidth=1)


