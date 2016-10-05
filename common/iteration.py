from common.classes.Struct import Struct
import numpy as np
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *


from itertools import tee, islice, chain




def piter(x, percent_period=1,period=None,end="| ", show=True):
    '''Iterates through x (any iterable object, having len) returning iteratively elements from x and printing progress.
    Progress is printed every <period> iterations or after every <percent_period> percent of total was complete.
    Useful for controlling how much of the long computations were completed.

    Example:
        for i in piter([10,11,12,13,14,15],2):
            print(i)
    Output:
        0.00% done
        10
        11
        33.33% done
        12
        13
        66.67% done
        14
        15
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    if show==False: # do nothing
        for element in x:
            yield element
    else:
    
        if hasattr(x,'__len__'):
            total = len(x)
            if period==None:
                period=max(1,total//(100/percent_period))
            for i,element in enumerate(x):
                if i % period==0:
                    print('%.0f' % (100*i/total), end=end)
                yield element
            print('100.0',end=end+'\n')
        else: # no len
            for i,element in enumerate(x):
                if i % period==0:
                    print('%d' % i, end=end)
                yield element
            print('%d'%i,end=end+'\n')




def iter_prev_cur_next(some_iterable):
    '''
    Iterate over elements yielding a tuple <previous element>,<current element>,<next element>.
    When there is no previous or next element returns None instead.
    
    EXAMPLE:
    mylist = ['banana', 'orange', 'apple', 'kiwi', 'tomato']
    for previous, item, nxt in prev_and_next(mylist):
        print ("Item is now", item, "next is", nxt, "previous is", previous)

    OUTPUT:
    Item is now banana next is orange previous is None
    Item is now orange next is apple previous is banana
    Item is now apple next is kiwi previous is orange
    Item is now kiwi next is tomato previous is apple
    Item is now tomato next is None previous is kiwi

    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.
    '''

    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)






def batches(iterable, batch_size = 10000):
    '''
    Iterates iterable by batches of size batch_size.

    EXAMPLE:
    for x in batches(list(range(10)), 3):
        print(x)

    PRINTS:
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''


    N = len(iterable)
    for i in range(0, N, batch_size):
       yield iterable[i:min(i+batch_size, N)]