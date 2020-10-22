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

fig, axes = plt.subplots(2,3)

# simulate once for a static, once for a changing switch
for i,da in enumerate([0, 0.3]):

    # setup the solver for the SSA
    S = ssa.OneEnzymeSSA(**p, epsilon=epsilon, kappa=kappa, a=a, da=da, bdeg=1, kX=kX, Xc=xc, omega=omega)

    S.simulate(200,dt=0.01)

    # after simulating, S has the attributes Xa and Xi which stand for the time series of active X and inactive X
    tv = S.tt
    Xv = S.Xa
    XTv = S.Xa + S.Xi
    if da > 0: # keep the a values for the changing switch, to use for plotting snapshots of the switch
        av = a + da*np.tanh(kappa*(XTv/omega - xc))

    axes[i,0].plot(XTv,Xv)
    axes[i,1].plot(tv, Xv, label='$X$')
    axes[i,1].plot(tv, XTv, label='$X_T$')

    #detect period using two thresholds
    # the function returns a list of tuples (time, up/down)
    ctt = csu.getcrossingtimes_twothresholds(tv, Xv, switch.folds[1][1]*omega, switch.folds[0][1]*omega)
    ctt2 = [] # new list with up and down jumps in turn

    j = 0

    # start with a new up jump
    while j < len(ctt):
        if ctt[j][1] != ctt[j-1][1] and ctt[j][1]=='u':
            ctt2.append(ctt[j][0])
            break
        j+=1
    j+=1
    # now take turns adding up and down jumps
    while j < len(ctt):
        if ctt[j][1] != ctt[j-1][1]:
            ctt2.append(ctt[j][0])

        j+=1
    ctt2 = np.array(ctt2)
    # extract the period as difference between the up jumps
    diffs = ctt2[2::2]-ctt2[:-2:2]
    permean = np.mean(diffs)
    persig = np.std(diffs)

    axes[i,2].hist(diffs)
    axes[i,2].axvline(permean,color='k')
    axes[i,2].set_title('Mean period: {:.2f}, Std period: {:.2f}'.format(permean,persig))

    # add thresholds to time series plot
    axes[i,1].axhline(switch.folds[0][1]*omega,color='k')
    axes[i,1].axhline(switch.folds[1][1]*omega,color='k')

# add labels etc
for ax in axes[:,0]: # phaseplanes
    ax.set_xlabel('$X_T$')
    ax.set_ylabel('$X$')
    ax.set_xlim(0,5*omega)
    ax.set_ylim(0,3*omega)

# plot the switch curve for static switch
axes[0,0].plot(switch.xtv*omega, switch.xv*omega, 'k')
# plot snapshots for the changing switch
# use a range of a values in between min and max a attained in the limit cycle
n=6
lowa = np.min(av[-500:])
higha = np.max(av[-500:])

for i,aa in enumerate(np.linspace(lowa,higha,n)):
    switch.a=aa
    switch.compute_responsecurve()
    axes[1,0].plot(switch.xtv*omega, switch.xv*omega, color='k', alpha=i/n)

for ax in axes[:,1]: # time series
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()
    ax.set_xlim(0,20) # take a selection

axes[0,0].set_title('Static')
axes[1,0].set_title('Changing')
plt.show()
