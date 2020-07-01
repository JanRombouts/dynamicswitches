## Demo file for the code that was used in the paper
## Dynamic bistable switches enhance robustness and accuracy of cell cycle transitions
## by Jan Rombouts and Lendert Gelens

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# import package to draw bistable switches and utility functions
import changingswitches as cs
import csutils as csu

# file for simulating front
import solveswitch as ss

# import package for simulation
from jitcsde import y as ys, t as ts, jitcsde

#set up parameters for the switch
p = dict(b=1., K=1, n=5, ap=.1, bp=1., Kp=1., m=5)
XT = 2



#### function to introdce kX in space and time -- NOT USED
def kXTf(x,t):
    return 0*np.ones_like(x)


### function af gives a in space and time
# the time argument can be used to release the front as in Animation 2
# here: ony space dependence

# we solve once for constant a and once for a changing from high to low such that the front gets stuck

# plots: on the left snapshots of the front, on the right the position as function of time
fig, axes = plt.subplots(2,3)

ab = 0.2 # mean value of a over space

for i,da in enumerate([0, 0.07]):

    def af(x,t):
        return ab - da*np.tanh(0.1*(x-50)) # transition is at x=50, quite smooth


    # set up the problem
    prob = ss.Problem(b=1., K=1, n=5, ap=.1, bp=1., Kp=1.,\
                      m=5, DX=5, DY=5, af=af, kXTf=kXTf, epsilon=1./20.)

    # initial condition
    lowX = 0.3
    highX = 1.8
    def initprofstandard(x):
        if x>10:
            return (lowX,XT-lowX)
        else:
            return (highX,XT-highX)

    solver = ss.Solver(prob,initprofstandard,T=20,dt=0.001,N=200,L=100,outputstep=50, \
                       bc='neumann', method='forwardeuler', nu=0.)

    solver.solve()
    #obtain the position of the front (= where a threshold is crossed)
    solver.getfrontposition(thr=1., flip=1)

    # plot the a profile
    axes[i,0].plot(solver.x, solver.problem.af(solver.x,0))

    # plot snapshots of the profile
    n = 5
    totsteps = solver.tt.shape[0] #total time points saved
    for j in range(0,totsteps,int(totsteps)//n):
        axes[i,1].plot(solver.x, solver.XX[j,:], alpha=(j+1)/totsteps)

    # front position
    axes[i,2].plot(solver.tt, solver.frontposition)

# labels etc
for ax in axes[:,0]:
    ax.set_xlabel('Position')
    ax.set_ylabel('$a$')
for ax in axes[:,1]:
    ax.set_xlabel('Position')
    ax.set_ylabel('$X$')
for ax in axes[:,2]:
    ax.set_xlabel('Time')
    ax.set_ylabel('Front position')
plt.show()
