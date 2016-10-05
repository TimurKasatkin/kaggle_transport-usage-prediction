# A set of distinct colors for visualization purposes.
# Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016

from pylab import *

# this is a set of well distinguishable colors. Useful for visualizing many graphs on one plot.
COLORS=[[0,0.5,1],[1,0,0],[0,0,0],[0.2,1,0],[1,0.5,0],[1,0,1],[0.5,0.5,0.5],[0.5,0,1],[1,1,0],[0,1,1],[ 0.25 ,  0.58,  0.50],[0,0,1]]



if __name__=='__main__':  # colors demonstration
    for i in range(len(COLORS)):
        plot([i,i],c=COLORS[i],linewidth=3)

    ylim([-1,len(COLORS)])