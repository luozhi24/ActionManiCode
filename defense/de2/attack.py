import numpy as np
import math
import random
from scipy import optimize

rho = 1/2
nu = 1
delta = 0.01
n = 10000000
###revise
#K = 0.9

"""
hctrho = 1/2
hctnu = 1
hctdelta  = 0.05
c = 2*math.sqrt(1/(1-hctrho))
"""

def f(x):
    return 0.5*(math.sin(13*x)*math.sin(27*x)+1)

class node:
    def __init__(self, h, i):
        self.h = h
        self.i = i
        self.L = float('-inf')
        self.B = float('-inf')
        self.sumreward = 0
        self.T = 0
        #self.tau = 0
        self.start = 0
        self.end = 0
        self.represent = -1
        self.leaf = 1

def iternode(h, i):
    return str(h) + "," + str(i)

"""
def tand(t):
    #return 2**(math.floor(math.log(t))+1)
    return 2**(math.floor(math.log2(t))+1)

def caldelta(t):
    return min((hctrho/(3*hctnu))**(1/8)*hctdelta/t, 1)

def caltau(t, h):
    return c**2*math.log(1/caldelta(tand(t)))*hctrho**(-2*h)/hctnu**2
"""

class attack:
    def __init__(self):
        self.attacktree = dict()
        self.T1 = dict()
        root = node(0,1)
        root.T = 1
        root.leaf = 0
        root.start = 0
        root.end = 1
        self.attacktree.update({iternode(0,1): root})
        self.attacktree.update({iternode(1,1): node(1,1)})
        self.attacktree.update({iternode(1,2): node(1,2)})
        self.attacktree[iternode(1,1)].start = 0
        self.attacktree[iternode(1,1)].end = 0.5
        self.attacktree[iternode(1,2)].start = 0.5
        self.attacktree[iternode(1,2)].end = 1

        self.T1.update({iternode(0,1): root})
        """
        self.T1.update({iternode(1,1): node(1,1)})
        self.T1.update({iternode(1,2): node(1,2)})
        self.T1[iternode(1,1)].start = 0
        self.T1[iternode(1,1)].end = 0.5
        self.T1[iternode(1,2)].start = 0.5
        self.T1[iternode(1,2)].end = 1
        """

        #self.nodenum = 3
        self.tm = 1
        self.reward = np.zeros(n+1)
        self.nodenum = np.zeros(n+1)
        self.nodenum[1] = 3
        self.hmax = 1
        self.brother = 0

    def caltau(self, h, i):
        nodeiter = iternode(h, i)
        sumT = self.attacktree[nodeiter].T
        return math.sqrt(1/(2*sumT)*math.log(math.pi**2*sumT**2*self.nodenum[self.tm]/(3*delta)))

    def calL(self, h ,i):
        nodeiter = iternode(h, i)
        return self.attacktree[nodeiter].sumreward/self.attacktree[nodeiter].T - nu*rho**h - self.caltau(h, i)

    def wortraverse(self):
        #tau = 0
        ht = 0
        it = 1
        nodeiter = iternode(ht, it)
        while self.attacktree[nodeiter].leaf == 0:# and nu*rho**ht >= tau:#self.caltau(ht, it):
            if self.attacktree[iternode(ht+1, 2*it-1)].B <= self.attacktree[iternode(ht+1, 2*it)].B:
                ht = ht+1
                it = 2*it-1
            else:
                ht = ht+1
                it = 2*it

            nodeiter = iternode(ht, it)
        return ht, it

    def updateB(self, ht, it):
        nodeiter = iternode(ht, it)
        if self.attacktree[nodeiter].leaf == 1:
            self.attacktree[nodeiter].B = self.attacktree[nodeiter].L
        else:
            self.attacktree[nodeiter].B = max(self.attacktree[nodeiter].L, min(self.attacktree[iternode(ht+1, 2*it-1)].B, self.attacktree[iternode(ht+1, 2*it)].B))
        
        h = ht - 1
        if it%2 == 0:
            i = int(it/2)
        else:
            i = int((it+1)/2)

        while h != 0:
            nodeiter = iternode(h, i)
            self.attacktree[nodeiter].B = max(self.attacktree[nodeiter].L, min(self.attacktree[iternode(h+1, 2*i-1)].B, self.attacktree[iternode(h+1, 2*i)].B))
            h = h-1
            if i % 2 == 0:
                i = int(i/2)
            else:
                i = int((i+1)/2)
        return

    def choose(self):
        if self.nodenum[self.tm] != self.nodenum[self.tm-1]:
            for key in self.attacktree.keys():
                if self.attacktree[key].h == 0:
                    continue
                if self.attacktree[key].T != 0:
                    h = self.attacktree[key].h
                    i = self.attacktree[key].i
                    self.attacktree[key].L = self.calL(h, i)

            for h in range(0, self.hmax):
                for key in self.attacktree.keys():
                    if self.attacktree[key].h == self.hmax - h:
                        if self.attacktree[key].leaf == 1:
                            self.attacktree[key].B = self.attacktree[key].L
                        else:
                            hnow = self.attacktree[key].h
                            inow = self.attacktree[key].i
                            self.attacktree[key].B = max(self.attacktree[key].L, min(self.attacktree[iternode(hnow+1, 2*inow-1)].B, self.attacktree[iternode(hnow+1, 2*inow)].B))
        
        ht, it = self.wortraverse()
        nodeiter = iternode(ht, it)
        if self.attacktree[nodeiter].represent == -1:
            self.attacktree[nodeiter].represent = self.attacktree[nodeiter].start + (self.attacktree[nodeiter].end - self.attacktree[nodeiter].start)*random.random()

        meanreward = f(self.attacktree[nodeiter].represent)
        self.reward[self.tm] = random.gauss(meanreward, 0.5)
        self.attacktree[nodeiter].T += 1
        self.attacktree[nodeiter].sumreward += self.reward[self.tm]
        self.attacktree[nodeiter].L = self.calL(ht, it)
        #self.tm += 1

        self.updateB(ht, it)
        #self.tm += 1
        if self.attacktree[nodeiter].leaf == 1 and nu*rho**ht >= self.caltau(ht, it):
            self.attacktree[nodeiter].leaf = 0
            self.attacktree.update({iternode(ht+1, 2*it-1): node(ht+1, 2*it-1)})
            self.attacktree[iternode(ht+1, 2*it-1)].start = self.attacktree[nodeiter].start
            self.attacktree[iternode(ht+1, 2*it-1)].end = (self.attacktree[nodeiter].start + self.attacktree[nodeiter].end)/2

            self.attacktree.update({iternode(ht+1,2*it): node(ht+1, 2*it)})
            self.attacktree[iternode(ht+1, 2*it)].start = (self.attacktree[nodeiter].start + self.attacktree[nodeiter].end)/2
            self.attacktree[iternode(ht+1, 2*it)].end = self.attacktree[nodeiter].end

            self.hmax = max(self.hmax, ht+1)
            self.tm += 1
            self.nodenum[self.tm] = self.nodenum[self.tm-1] + 2
        else:
            self.tm += 1
            self.nodenum[self.tm] = self.nodenum[self.tm-1]

        return self.reward[self.tm-1]
"""
    def judge(self, xt, t, k):
        h = 0
        i = 1
        #tau = 1
        flag = 0
        attackornot = 0

        nodeiter = iternode(h, i)
        print(k)
        print(delta)

        if self.brother:
            for key in self.T1.keys():
                if self.T1[key].h == 0:
                    continue
                if self.T1[key].represent == -1:
                    self.T1[key].represent = xt
                    nodeiter = key
                    flag = 1
                    self.brother = 0
                    break
        
        if flag == 0:
            for key in self.T1.keys():
                if xt == self.T1[key].represent:
                    flag = 1
                    nodeiter = key
                    break

        if flag == 0:
            for key in self.T1.keys():
                if self.T1[key].leaf == 1:
                    if xt >= self.T1[key].start and xt < self.T1[key].end:
                        nodeiter = key
                        break
            
            h = self.T1[nodeiter].h
            i = self.T1[nodeiter].i
            self.T1[nodeiter].leaf = 0
            self.T1.update({iternode(h+1, 2*i-1): node(h+1, 2*i-1)})
            self.T1[iternode(h+1, 2*i-1)].start = self.T1[iternode(h, i)].start
            self.T1[iternode(h+1, 2*i-1)].end = (self.T1[iternode(h, i)].start + self.T1[iternode(h, i)].end) / 2
            if xt >= self.T1[iternode(h+1, 2*i-1)].start and xt < self.T1[iternode(h+1, 2*i-1)].end:
                nodeiter = iternode(h+1, 2*i-1)

            self.T1.update({iternode(h+1, 2*i): node(h+1, 2*i)})
            self.T1[iternode(h+1, 2*i)].start = (self.T1[iternode(h, i)].start + self.T1[iternode(h, i)].end) / 2
            self.T1[iternode(h+1, 2*i)].end = self.T1[iternode(h, i)].end
            if xt >= self.T1[iternode(h+1, 2*i)].start and xt < self.T1[iternode(h+1, 2*i)].end:
                nodeiter = iternode(h+1, 2*i)

            self.T1[nodeiter].represent = xt
            #self.nodenum += 2
            self.brother = 1

        if k >= self.T1[nodeiter].start and k < self.T1[nodeiter].end:
            attackornot = 0
        else:
            attackornot = 1

        if attackornot:
            return self.choose(), attackornot, self.T1[nodeiter].h, self.T1[nodeiter].i
        else:
            meanreward = f(xt)
            return random.gauss(meanreward, 0.5), attackornot, self.T1[nodeiter].h, self.T1[nodeiter].i
"""   
        
    
        
        
