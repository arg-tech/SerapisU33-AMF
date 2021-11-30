
import re
import numpy as np
from nltk import ngrams
from collections import defaultdict


    
def top_freq_list(xs, top):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]

  
def frequent_tuple(tuples):
    count_tuple={}
    for tup in tuples:
        #print('tup..............................................',tup)
        length=len(tup[0].split(" "))
        if length in count_tuple.keys():
            count_tuple[length]+=1
        else:
            count_tuple[length]=1
    sorted_dct=dict(sorted(count_tuple.items(), reverse=True,key=lambda item: item[1]))
    return next(iter( sorted_dct.items() ))[0] 
    
    

def identyfy_maxs_index(x,bar): 
    return x > bar

