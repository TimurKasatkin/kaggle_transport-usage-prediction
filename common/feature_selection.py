from pylab import *

def forward_search(loss_fun, all_features, current_features=None, current_loss=None,
                   max_iterations=inf, accuracy=0.0001, par=None):
    '''Feature selection using forward search algorithm.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    print('FORWARD SEARCH:')
    if current_features is None:
        current_features=[]
        current_loss = inf

    iteration=1
    while iteration<=max_iterations:
        features = [feature for feature in all_features if feature not in current_features]
        current_features_candidates = [current_features+[feature] for feature in features]
        if par is None: # no parallelization
            losses = list( map(loss_fun, current_features_candidates) )
        else:
            losses = list( par.map(loss_fun, current_features_candidates) )
        i = argmin(losses)
        if losses[i]<current_loss-accuracy:
            print('%30s: delta_loss=%.3f new_loss=%.3f'%('+%s'%features[i], current_loss-losses[i], losses[i]) )
            current_features = current_features_candidates[i]
            current_loss = losses[i]
        else:
            print('Next min loss=%.3f. Stopping.'%losses[i])
            break
        iteration+=1

    return current_features, current_loss



def backward_search(loss_fun, current_features, current_loss=None, accuracy=0.0001, par=None):
    '''Feature selection using backward search algorithm.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    print('BACKWARD SEARCH:')
    if len(current_features)<=1:
        print('#features<=1. No exclusions possible.')
        return current_features, current_loss

    if current_loss is None:
        current_loss = loss_fun(current_features)
        print('Current loss: %.3f' % current_loss)

    while True:
        current_features_candidates = []
        for feature in current_features:
            candidate = current_features.copy()
            candidate.remove(feature)
            current_features_candidates.append( candidate )

        if par is None: # no parallelization
            losses = list( map(loss_fun, current_features_candidates) )
        else:
            losses = list( par.map(loss_fun, current_features_candidates) )
        i = argmin(losses)
        if losses[i]<current_loss-accuracy:
            print('%30s: delta_loss=%.3f new_loss=%.3f'%('-%s'%current_features[i], current_loss-losses[i], losses[i]) )
            current_features = current_features_candidates[i]
            current_loss = losses[i]
        else:
            print('Next min loss=%.3f. Stopping.'%losses[i])
            return current_features, current_loss



def forward_backward_search(loss_fun, all_features, current_features=None, current_loss=None, accuracy=0.001, par=None):
    '''Feature selection using forward-backward search algorithm.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    if (current_features is None and current_loss is None):
        current_features=[]
        current_loss=inf
    elif current_loss is None and current_features is not None:
        current_loss = loss_fun(current_features)
        print('Current loss: %.3f' % current_loss)
    else:
        assert (current_features is not None) and (current_loss is not None)


    while True:
        before_change_loss = current_loss

        new_current_features, new_current_loss = forward_search(loss_fun, all_features, current_features,
                                                                current_loss, 1, accuracy, par)
        if new_current_loss<current_loss:
            current_features, current_loss = new_current_features, new_current_loss

        new_current_features, new_current_loss = backward_search(loss_fun, current_features, current_loss, accuracy, par)
        if new_current_loss<current_loss:
            current_features, current_loss = new_current_features, new_current_loss

        if (before_change_loss == current_loss):
            return current_features, current_loss