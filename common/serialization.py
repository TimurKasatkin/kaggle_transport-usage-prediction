#!/usr/bin/env python
# encoding: utf-8
# Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016


import pickle


def pickle_save(Object,rFilename,protocol=-1):
    '''Save object to file.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''

    f = open(rFilename,'wb')
    if protocol is None:
        pickle.dump(Object,f)
    else:
        pickle.dump(Object,f,protocol=protocol)
    f.close()

    
def pickle_load(rFilename):
    '''Load object from file.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
    
    f=open(rFilename,'rb')
    Object = pickle.load(f)
    f.close()
    return Object
