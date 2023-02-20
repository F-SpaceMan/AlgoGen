from numpy.random import uniform
from numpy.random import randint
from numpy.random import rand
import numpy as np
get_ipython().run_line_magic('matplotlib', 'qt')

import matplotlib.pyplot as plt

def fobjetivo(individuo1, individuo2):
    x = individuo1
    y = individuo2
    f = np.sin(x)*np.exp((1-np.cos(y))**2) + np.cos(y)*np.exp((1-np.sin(x))**2)+ (x-y)**2
    return f


def selecao(P, fitness, N=3):

    selecao_i = randint(len(P))
    for i in randint(0, len(P), N-1):

        if fitness[i] < fitness[selecao_i]:
            selecao_i = i
    return P[selecao_i]


def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    
    if rand() < r_cross:
        kth = randint(0, len(p1))
        c1 = p1[:kth] + [(1-r_cross) * p1[kth] + r_cross * p2[kth]] + p1[kth+1:]
        c2 = p2[:kth] + [(1-r_cross) * p2[kth] + r_cross * p1[kth]] + p2[kth+1:]
    return [c1, c2]


def muta(c, r_mut, intervalo):
    
    for i in range(len(c)):
        if rand() < r_mut:
            if i == 0:
                c[i] = uniform(intervalo[0][0], intervalo[0][1])
            else:
                c[i] = uniform(intervalo[1][0], intervalo[1][1])
                
            

# genetic algorithm
def GA(fobjetivo, intervalo, numGer, popsize, r_cross, r_mut):
    pop = []

    for _ in range (popsize):
        xi = uniform(intervalo[0][0], intervalo[0][1])
        yi = uniform(intervalo[1][0], intervalo[1][1])

        pop.append([xi, yi])
    
    best = 0
    best_eval = 0
    
    
    X, Y = [], []
    Z = []
    
    
    for gen in range(numGer):
        
        individuos = []

        for p in pop:
            individuos.append(p)

        fitness = [fobjetivo(d[0], d[1]) for d in individuos]
        selected = []
        

        for i in range(popsize):
            if fitness[i] < best_eval:
                best, best_eval = pop[i], fitness[i]
    
        for _ in range(popsize):
            selected.append(selecao(pop, fitness))

        children = []
        
        for i in range(0, popsize, 2):
            p1, p2 = selected[i], selected[i+1]

            for c in crossover(p1, p2, r_cross):
                muta(c, r_mut, intervalo)
                children.append(c)

        pop = children
        
        X.append(best[0])
        Y.append(best[1])
        Z.append(best_eval)
        
    return [best, best_eval, X, Y, Z]


intervalo = [[-10.0, 10.0], [-10.0, 10.0]]

popsize = 2000
numGer = 1000
crossrate = 0.7
mutrate = 0.001


X, Y, Xarr, Yarr, Zarr = GA(fobjetivo, intervalo, numGer, popsize, crossrate, mutrate)
print('f(%s) = %f' % (X, Y))

fig = plt.figure()
    
fig.set_figwidth(10)
fig.set_figheight(10)

plt.title("XY-Geração")
plt.plot(np.array(Xarr), label='x')
plt.plot(np.array(Yarr), label='y')
plt.xlabel("Geração") 
plt.ylabel("Valor") 
plt.legend() 

plt.show()

fig = plt.figure()

ax = plt.axes(projection ="3d")
ax.grid(visible = True, linewidth = 0.9, alpha = 0.4)

my_cmap = plt.get_cmap('hsv')
sctt = ax.scatter3D(Xarr, Yarr, Zarr,
                alpha = 0.2,
                marker = 'o',
                s = 500,
                c = 'g')
plt.title("XYZ-Histórico")
fig.set_figwidth(10)
fig.set_figheight(10)

plt.show()

