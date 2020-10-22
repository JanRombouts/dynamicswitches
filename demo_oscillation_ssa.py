## Demo file for the code that was used in the paper
## Dynamic bistable switches enhance robustness and accuracy of cell cycle transitions
## by Jan Rombouts and Lendert Gelens

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# import package to draw bistable switches and utility functions
import changingswitches as cs
import csutils as csu

# import package for simulation of the Gillespie algorithm
import oneenzymessa as ssa

## Simulate
p = dict(b=1., K=1, n=5, ap=0.1, bp=1., Kp=1., m=5)

a = 0.3

# compute switch
switch= cs.ResponseOneEnzymeXt(a=a,**p)
switchdat = []

switch.setcontpars(0.01,2000)
switch.setstart(0,0,[0.1,0.1])
switch.compute_responsecurve()


kX = 1.7
kappa = 5.
epsilon = 0.05

omega = 50 # system size, correlates with number of molecules

# threshold value used in a(X_T)
xc = 0.5*(switch.folds[0][0] + switch.folds[1][0])

fig, axes = plt.subplots(2,2)

# simulate once for a static, once for a changing switch
for i,da in enumerate([0, 0.3]):

    # setup the solver for the SSA
    S = ssa.OneEnzymeSSA(**p, epsilon=epsilon, kappa=kappa, a=a, da=da, bdeg=1, kX=kX, Xc=xc, omega=omega)

    S.simulate(20,dt=0.01)

    # after simulating, S has the attributes Xa and Xi which stand for the time series of active X and inactive X
    tv = S.tt
    Xv = S.Xa
    XTv = S.Xa + S.Xi

    axes[i,0].plot(XTv,Xv)
    axes[i,1].plot(tv, Xv, label='$X$')
    axes[i,1].plot(tv, XTv, label='$X_T$')

    #detect period using two thresholds
    #thresholds are up and down
    ctt = csu.getcrossingtimes_twothresholds(tv, Xv, switch.folds[1][1]*omega, switch.folds[0][1]*omega)
    ctt2 = [] # take turns up down up
    j = 0
    # start with an up jump always
    while j < len(ctt):
        if ctt[j][1] != ctt[j-1][1] and ctt[j][1]=='u':
            ctt2.append(ctt[j][0])
            break
        j+=1
    j+=1
    while j < len(ctt):
        if ctt[j][1] != ctt[j-1][1]:
            ctt2.append(ctt[j][0])
        j+=1
    ctt2 = np.array(ctt2)
    # at this point, ctt2 contains time coordinates of up, down, up down jumps.
    # period is average difference between two up jumps, for example

    if len(ctt2)>3:
        per2 = np.mean(np.diff(ctt2[::2]))
    else:
        per2=0


    axes[i,1].set_title('Period: {:.3f}'.format(per2))

    if da > 0: # keep the a values for the changing switch, to use for plotting snapshots of the switch
        av = a + da*np.tanh(kappa*(XTv/omega-xc))
# add labels etc
for ax in axes[:,0]: # phaseplanes
    ax.set_xlabel('$X_T$')
    ax.set_ylabel('$X$')
    ax.set_xlim(0,5*omega)
    ax.set_ylim(0,3*omega)

# plot the switch curve for static switch
# note: multiply by omega
axes[0,0].plot(switch.xtv*omega, switch.xv*omega, 'k')

# draw some snapshots for the changing switch
n=6
lowa = np.min(av[-500:])
higha = np.max(av[-500:])

for i,a in enumerate(np.linspace(lowa,higha,n)):
    switch.a=a
    switch.compute_responsecurve()
    axes[1,0].plot(switch.xtv*omega, switch.xv*omega, color='k', alpha=i/n)

for ax in axes[:,1]: # time series
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()

axes[0,0].set_title('Static')
axes[1,0].set_title('Changing')



plt.show()
