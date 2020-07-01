## Demo file for the code that was used in the paper
## Dynamic bistable switches enhance robustness and accuracy of cell cycle transitions
## by Jan Rombouts and Lendert Gelens

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# import package to draw bistable switches and utility functions
import changingswitches as cs
import csutils as csu

# import package for simulation
from jitcsde import y as ys, t as ts, jitcsde

#set up parameters for the switch
p = dict(ap=0.1, b=1, bp=1, K=1, Kp=1, n=5, m=5)

# Hill functions

def f(x,a):
    return a +p['b']*x**p['n']/(p['K']**p['n'] + x**p['n'])
def g(x):
    return p['ap']+p['bp']*p['Kp']**p['m']/(p['Kp']**p['m'] + x**p['m'])

a=0.3
# setup switch
switch = cs.ResponseOneEnzymeXt(a=a, **p)
switch.setcontpars(0.01,1000)
switch.setstart(0,0,[1,0])
switch.compute_responsecurve()

# simulate noisy system
sigma = 0.4
kX = .2

#set up functions for the changing switch
#threshold for a(X_T)
Xc = 0.5*(switch.folds[0][0] + switch.folds[1][0])
kappa = 5

def F(X):
    return sp.tanh(kappa*(X-Xc))
def Fn(X):
    return np.tanh(kappa*(X-Xc))


fig, axes = plt.subplots(2, 2)

# we solve twice, once for static switch (da=0) and once for changing switch

for i,da in enumerate([0, 0.3]):

    dxdt = [20*(f(ys(0),a + da*F(ys(1)))*(ys(1)-ys(0)) - g(ys(0))*ys(0)), kX]
    ggg = [sigma, 0] # noise term: only noise in X variable
    ini = [0.01,0.01]

    sdesys = jitcsde(dxdt, ggg)

    sdesys.set_initial_value(ini,0.0)
    timeseries = []

    tv = np.arange(sdesys.t, sdesys.t+5/kX, 0.01)
    for time in tv:
        timeseries.append( sdesys.integrate(time) )
    timeseries = np.array(timeseries)
    Xv = timeseries[:,0]
    XTv = timeseries[:,1]

    # detect transition timing
    # threshold: middle of vertical coordinates of folds
    thrcross = 0.5*(switch.folds[0][1]+switch.folds[1][1])
    cts = csu.getcrossingtimes(tv,Xv,thrcross) # utility function
    ct = cts[0] # cts is a list
    ## plots: left phaseplane, right time series
    axes[i,0].plot(switch.xtv, switch.xv, 'k')
    axes[i,0].plot(XTv,Xv)

    axes[i,1].plot(tv, Xv, label='$X$')
    axes[i,1].plot(tv, XTv, label='$X_T$')

    # add the crossing time
    axes[i,1].plot([ct, ct, 0],[0, thrcross, thrcross],'k--')

##############################################################

# add labels etc
for ax in axes[:,0]: # phaseplanes
    ax.set_xlabel('$X_T$')
    ax.set_ylabel('$X$')

# plot the switch curve for static switch
axes[0,0].plot(switch.xtv, switch.xv, 'k')

# plot snapshots for the changing switch
n=6
for i,a in enumerate(np.linspace(a-da,a+da,n)):
    switch.a=a
    switch.compute_responsecurve()
    axes[1,0].plot(switch.xtv, switch.xv, color='k', alpha=i/n)

for ax in axes[:,1]: # time series
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()

axes[0,0].set_title('Static')
axes[1,0].set_title('Changing')
plt.show()
