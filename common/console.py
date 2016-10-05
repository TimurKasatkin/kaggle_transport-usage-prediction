#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from time import time

from pylab import *





class Logger:
    '''Logger is able to print all messages of level above threshold specified by user.
    Message level specifies message unimprotance: the lower the level - the more important is the message.
    This helps to control the detalization of output.
    
    Params:
        level (int): threshold level. All messages with level <= threshold are printed.
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''
        
    def __init__(self,level):
        self.level=level

    def pr0(self,s,*args,**nargs):
        if self.level>=0:
            print(s,*args,**nargs)        
        
    def pr1(self,s,*args,**nargs):
        if self.level>=1:
            print(s,*args,**nargs)

    def pr2(self,s,*args,**nargs):
        if self.level>=2:
            print(s,*args,**nargs)

    def pr3(self,s,*args,**nargs):
        if self.level>=3:
            print(s,*args,**nargs)


