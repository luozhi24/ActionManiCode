import numpy as np
import math
import random
import matplotlib.pyplot as plt
from numpy.core.defchararray import count 
#import attack
from scipy import optimize
import time
import sys

rho = 1/2
nu = 1 #math.sqrt(2)
deltal = 0.05
c = 2*math.sqrt(1/(1-rho))
n = 10000000
hmax = 1
random.seed()
#K = 0.0
#att = attack.attack()

class node:
    def __init__(self, h, i):
        self.h = h
        self.i = i
        self.U = float('inf')
        self.B = float('inf')
        self.sumreward = 0
        self.T = 0
        #self.tau = 0
        self.start = 0
        self.end = 0
        self.represent = -1
        self.leaf = 1

def f(x):
    return 0.5*(math.sin(13*x)*math.sin(27*x)+1)

def oppof(x):
    return -1*0.5*(math.sin(13*x)*math.sin(27*x)+1)

def tand(x):
    #return 2**(math.floor(math.log(t))+1)
    return 2**(math.floor(math.log(x))+1)

def delta(x):
    return min((rho/(3*nu))**(1/8)*deltal/x, 1)

def iternode(h, i):
    return str(h) + "," + str(i)

def caltau(x, h):
    return c**2*math.log(1/delta(tand(x)))*rho**(-2*h)/nu**2

opti = optimize.minimize(oppof, 0.85, method='SLSQP')
maxv = -1 * opti.fun
print(maxv)

#optix = optimize.minimize(oppof, 0.4, method='SLSQP')
bestarm = opti.x[0]
print(bestarm)
time.sleep(5)

maintree = dict()
root = node(0, 1)
root.T = 1
root.leaf = 0
root.start = 0
root.end = 1
#root.tau = 1
#print(iternode(0, 1))
maintree.update({iternode(0, 1): root})
maintree.update({iternode(1, 1): node(1, 1)})
maintree.update({iternode(1, 2): node(1, 2)})
maintree[iternode(1, 1)].start = 0
maintree[iternode(1, 1)].end = 0.5
maintree[iternode(1, 2)].start = 0.5
maintree[iternode(1, 2)].end = 1

#print(maintree[iternode(1, 2)].i)

def opttraverse(t):
    tau = 1
    ht = 0
    it = 1
    nodeiter = iternode(ht, it)
    while maintree[nodeiter].leaf == 0 and maintree[nodeiter].T >= tau:
        if maintree[iternode(ht+1, 2*it-1)].B >= maintree[iternode(ht+1, 2*it)].B:
            ht = ht+1
            it = 2*it-1
        else:
            ht = ht+1
            it = 2*it
        tau = caltau(t, ht)
        #print(maintree[nodeiter].T, tau)
        nodeiter = iternode(ht, it)
    return ht, it

def updateB(ht, it):
    nodeiter = iternode(ht, it)
    if maintree[nodeiter].leaf == 1:
        maintree[nodeiter].B = maintree[nodeiter].U
    else:
        maintree[nodeiter].B = min(maintree[nodeiter].U, max(maintree[iternode(ht+1, 2*it-1)].B, maintree[iternode(ht+1, 2*it)].B))

    h = ht - 1
    if it % 2 == 0:
        i = int(it/2)
    else:
        i = int((it+1)/2)

    while h != 0:
        nodeiter = iternode(h, i)
        maintree[nodeiter].B = min(maintree[nodeiter].U, max(maintree[iternode(h+1, 2*i-1)].B, maintree[iternode(h+1, 2*i)].B))
        h = h-1
        if i % 2 == 0:
            i = int(i/2)
        else:
            i = int((i+1)/2)
    return

def calU(treekey, time):
    return maintree[treekey].sumreward/maintree[treekey].T + nu * rho**(maintree[treekey].h) + math.sqrt(c**2*math.log(1/delta(tand(time)))/maintree[treekey].T)

print("HCT under no attack")
t = 1
reward = np.zeros(n+1)
sumreward = np.zeros(n+1)
regret = np.zeros(n+1)
timelist = np.zeros(n+1)
cost = np.zeros(n+1)
countcost = np.zeros(n+1)
target = np.zeros(n+1)
nonbest = np.zeros(n+1)
while t <= n:
    if t == tand(t):
        for key in maintree.keys():
            if maintree[key].h == 0:
                continue
            if maintree[key].T != 0:
                maintree[key].U = calU(key, t)

        for h in range(0, hmax):
            for key in maintree.keys():
                if maintree[key].h == hmax - h:
                    if maintree[key].leaf == 1:
                        maintree[key].B = maintree[key].U
                    else:
                        hnow = maintree[key].h
                        inow = maintree[key].i
                        maintree[key].B = min(maintree[key].U, max(maintree[iternode(hnow+1, 2*inow-1)].B, maintree[iternode(hnow+1, 2*inow)].B))

    ht, it = opttraverse(t)
    print("choose node", ht, it)
    nodeiter = iternode(ht, it)
    #print("choose node meanreward:", maintree[nodeiter].sumreward/maintree[nodeiter].T)
    if maintree[nodeiter].represent == -1:
        maintree[nodeiter].represent = maintree[nodeiter].start + (maintree[nodeiter].end - maintree[nodeiter].start)*random.random()

    print("start: ", maintree[nodeiter].start)
    print("end: ", maintree[nodeiter].end)
    print("represent: ", maintree[nodeiter].represent)
    if bestarm >= maintree[nodeiter].start and bestarm < maintree[nodeiter].end:
        print("choose best!")
        nonbest[t] = nonbest[t-1]
    else:
        nonbest[t] = nonbest[t-1] + 1
    
    #reward[t], attackornot, tcost, hj, ij = att.judge(maintree[nodeiter].represent, t, K)
    
    tcost = 0
    attackornot = 0
    meanreward = f(maintree[nodeiter].represent)
    reward[t] = random.gauss(meanreward, 0.5)
    

    """
    if hj != ht or ij != it:
        print("judge node error!\n")
        exit(0)
    """
    
    """
    if attackornot:
        cost[t] = cost[t-1] + tcost
        countcost[t] = countcost[t-1] + 1
        target[t] = target[t-1]
    else:
        cost[t] = cost[t-1]
        countcost[t] = countcost[t-1]
        target[t] = target[t-1] + 1
    """

    sumreward[t] = sumreward[t-1] + reward[t]
    regret[t] = t * maxv - sumreward[t]
    timelist[t] = t
    maintree[nodeiter].T += 1
    maintree[nodeiter].sumreward += reward[t]
    t += 1
    maintree[nodeiter].U = calU(nodeiter, t)
    #print(maintree[nodeiter].U)
    print("choose node meanreward:", maintree[nodeiter].sumreward/maintree[nodeiter].T)

    updateB(ht, it)
    #print(maintree[nodeiter].T, caltau(t, ht))
    print()
    if maintree[nodeiter].T >= caltau(t, ht) and maintree[nodeiter].leaf == 1:
        maintree[nodeiter].leaf = 0
        maintree.update({iternode(ht+1, 2*it-1): node(ht+1, 2*it-1)})
        maintree[iternode(ht+1, 2*it-1)].start = maintree[iternode(ht, it)].start
        maintree[iternode(ht+1, 2*it-1)].end = (maintree[iternode(ht, it)].start + maintree[iternode(ht, it)].end) / 2

        maintree.update({iternode(ht+1, 2*it): node(ht+1, 2*it)})
        maintree[iternode(ht+1, 2*it)].start = (maintree[iternode(ht, it)].start + maintree[iternode(ht, it)].end) / 2
        maintree[iternode(ht+1, 2*it)].end = maintree[iternode(ht, it)].end

        hmax = max(hmax, ht+1)
"""
plt.plot(timelist, countcost)
plt.title('Attack times')
plt.show()
plt.plot(timelist, target)
plt.title('Target arm pulled')
plt.show()
"""

'''
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1) # 画1行2列个图形的第1个
ax2 = fig.add_subplot(1,2,2) # 画1行2列个图形的第2个

ax1.plot(timelist, regret)
ax2.plot(timelist, attsum)
plt.show()
'''

"""
#print(hmax)
plt.plot(timelist, regret)
plt.show()
"""
fpregret = open("./infor" + sys.argv[1] + "/regret","w")
fpregret.write("round:" + str(sys.argv[1]) + "\n")
for i in range(len(regret)):
    fpregret.write(str(regret[i]))
    fpregret.write(" ")
fpregret.write("\n")
fpregret.close()






