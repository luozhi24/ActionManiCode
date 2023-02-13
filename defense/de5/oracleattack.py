import numpy as np
import math
import random
from scipy import optimize

"""
hctrho = 1/2 
hctnu = 1
hctdelta  = 0.05
c = 2*math.sqrt(1/(1-hctrho))
"""

def f(x):
    return 0.5*(math.sin(13*x)*math.sin(27*x)+1)

class attack:
    def __init__(self):
        worti = optimize.minimize(f, 0.65, method='SLSQP')
        self.minv = worti.fun
        print("oracleattack.py")


    def choose(self):
        #print("oracle minv:", self.minv)
        return random.gauss(self.minv, 0.5)



        
        
