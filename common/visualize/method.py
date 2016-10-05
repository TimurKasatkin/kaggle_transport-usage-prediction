#!/usr/bin/env python
# encoding: utf-8

from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import tree
from sklearn.metrics import accuracy_score
from common.iteration import piter
from common.functions import vec, normalize
from pylab import *



def show_param_dependency(clf, X_train, Y_train, param_name, param_vals, x_label=None, score_fun='accuracy'):
    '''Show plot, showing dependency of score_fun (estimated using CV on X_train, Y_train) 
       on parameter param_name taking values in param_vals.
       Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
       
    
    if x_label is None:
        x_label = param_name
        
    scores = zeros(len(param_vals))

    for val_num, param_val in enumerate(piter(param_vals)):
        clf.set_params(**{param_name:param_val})
        clf_cv = skl.grid_search.GridSearchCV(clf,param_grid={},scoring=score_fun,n_jobs=1, refit=False)
        clf_cv.fit(X_train, Y_train)
        scores[val_num] = clf_cv.best_score_

    xlabel(x_label)
    ylabel(score_fun)
    plot(param_vals, scores)
    print('Top %s = %.4f for %s=%s' % (score_fun, max(scores), param_name, param_vals[argmax(scores)]) )  
    

    


def print_decision_tree(tree, feature_names=None, class_names=None, offset_unit='    '):
    '''Plots textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1,f2,f3,... if not specified
    offset_unit: a string of offset of the conditional block
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    if tree.tree_.n_classes[0]==1:
        task = 'regression'
    else:
        task = 'classification'
    
    if feature_names is None:
        features  = ['f%d'%i for i in tree.tree_.feature]
    else:
        features  = [feature_names[i] for i in tree.tree_.feature]  

    def recurse(task, left, right, threshold, features, node, depth=0):
            offset = offset_unit*depth
            if (threshold[node] != -2):
                    print(offset+"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                    if left[node] != -1:
                            recurse (task, left, right, threshold, features,left[node],depth+1)
                    print(offset+"} else {")
                    if right[node] != -1:
                            recurse (task, left, right, threshold, features,right[node],depth+1)
                    print(offset+"}")
            else:
                    if task=='regression':
                        print(offset+"return %s" % value[node][0,0] )
                    else: # classification
                        y = argmax(value[node][0])
                        if class_names is None:
                            print(offset+"return class%d"%y)
                        else:
                            print(offset+"return " + class_names[y])

    recurse(task, left, right, threshold, features, 0,0)

    
    
    
def visualize_tree(clf,filename):
    '''Writes visualization of tree clf into pdf file filename.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    

    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename)


    
def feature_importances(clf,features):
    '''Shows feature importances in descending order for predictor clf. Feature names are given in list features.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    importances = clf.feature_importances_
    inds = np.argsort(importances)[::-1]
    inds = inds[importances>0]
    ordered_features = [features[ind] for ind in inds]
    ordered_importances = importances[inds]
    for feature, importance in zip(ordered_features, ordered_importances):
        print('%35s: %.4f' % (feature, importance))
    return (ordered_features, ordered_importances)

    
    
def gb_error_metric_plot(est, X_train,y_train, X_test, y_test, ax=None, label='', train_color=[0,1,0],
                  test_color=[1,0,0], alpha=1.0):
    '''Plots sequential error rate for each step of gradient boosted trees.
       Returs handle to axis where the plot was shown.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
       

    fun = lambda y_test,y_pred: 1-accuracy_score(y_test,y_pred)

    n_estimators = est.get_params()['n_estimators']
    err_train = np.zeros(n_estimators)
    err_test = np.zeros(n_estimators)

    for i, (pred_train,pred_test) in enumerate(zip(est.staged_predict(X_train),est.staged_predict(X_test))):
        err_train[i] = fun(y_train, pred_train)
        err_test[i] = fun(y_test, pred_test)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

    ax.plot(np.arange(n_estimators) + 1, err_train, color=train_color,
             label='Train', linewidth=2, alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1, err_test, color=test_color,
             label='Test', linewidth=2, alpha=alpha)
    ax.set_ylabel('Error rate')
    ax.set_xlabel('n_estimators')
    ax.set_ylim((0, 1.1*max(err_train.max(),err_test.max())))
    ax.legend()
    title = '{}: min_train={:.04f}, min_test={:.04f}'.format(label,err_train.min(),err_test.min())
    ax.set_title(title)
    return ax
