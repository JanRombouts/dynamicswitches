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
from jitcdde import y as yd, t as td, jitcdde

## Simulate
p = dict(b=1., K=1, n=5, ap=0.1, bp=1., Kp=1., m=5)

a = 0.3

# Hill functions
def f(x,a):
    return a +p['b']*x**p['n']/(p['K']**p['n'] + x**p['n'])
def g(x):
    return p['ap']+p['bp']*p['Kp']**p['m']/(p['Kp']**p['m'] + x**p['m'])



# compute switch
switch= cs.ResponseOneEnzymeXt(a=a,**p)
switchdat = []

switch.setcontpars(0.01,2000)
switch.setstart(0,0,[0.1,0.1])
switch.compute_responsecurve()


#Simulation for a system without noise, with possible time delay

tau = 0.1
kX = 1.6
kappa = 5.
epsilon = 0.05

# threshold value used in a(X_T)
Xc = 0.5*(switch.folds[0][0] + switch.folds[1][0])

def F(x, xc):
    #function which returns values between -1 and 1
    return sp.tanh(kappa*(x-xc))
def Fn(x, xc):
    return np.tanh(kappa*(x-xc))

fig, axes = plt.subplots(2,2)

# simulate once for a static, once for a changing switch
for i,da in enumerate([0, 0.3]):
    dxdt = [1/epsilon*(f(yd(0), a+da*F(yd(1,td-tau), Xc))*(yd(1)-yd(0))- g(yd(0))*yd(0)), kX-yd(1)*yd(0)]
    y0 = [0,0]
    ddesys = jitcdde(dxdt)
    ddesys.constant_past(y0)
    ddesys.step_on_discontinuities(max_step=0.01)
    ddesys.set_integration_parameters(first_step=0.001,max_step=0.01)

    timeseries = []

    tv = np.arange(ddesys.t, ddesys.t+20, 0.01)
    for time in tv:
        timeseries.append( ddesys.integrate(time) )

    timeseries = np.array(timeseries)
    Xv = timeseries[:,0]
    XTv = timeseries[:,1]
    if da > 0: # keep the a values for the changing switch, to use for plotting snapshots of the switch
        av = a + da*Fn(XTv,Xc)

    axes[i,0].plot(XTv,Xv)
    axes[i,1].plot(tv, Xv, label='$X$')
    axes[i,1].plot(tv, XTv, label='$X_T$')

    #detect period using two thresholds
    #thresholds are up and down
    ct = csu.getcrossingtimes_twothresholds(tv, Xv, switch.folds[1][1], switch.folds[0][1])
    if len(ct)>3:
        per2 = abs(ct[-1][0]-ct[-3][0])
    else:
        per2=0
    axes[i,1].set_title('Period: {:.3f}'.format(per2))


# add labels etc
for ax in axes[:,0]: # phaseplanes
    ax.set_xlabel('$X_T$')
    ax.set_ylabel('$X$')
    ax.set_xlim(0,5)
    ax.set_ylim(0,3)

# plot the switch curve for static switch
axes[0,0].plot(switch.xtv, switch.xv, 'k')

# plot snapshots for the changing switch
# use a range of a values in between min and max a attained in the limit cycle
n=6
lowa = np.min(av[-500:])
higha = np.max(av[-500:])

for i,a in enumerate(np.linspace(lowa,higha,n)):
    switch.a=a
    switch.compute_responsecurve()
    axes[1,0].plot(switch.xtv, switch.xv, color='k', alpha=i/n)

for ax in axes[:,1]: # time series
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()

axes[0,0].set_title('Static')
axes[1,0].set_title('Changing')

##########################################################################################

# System with noise, no time delay

kX = 1.6
kappa = 5.
epsilon = 0.05
sigma = 0.2

# threshold value used in a(X_T)
Xc = 0.5*(switch.folds[0][0] + switch.folds[1][0])

def F(x, xc):
    #function which returns values between -1 and 1
    return sp.tanh(kappa*(x-xc))
def Fn(x, xc):
    return np.tanh(kappa*(x-xc))

fig, axes = plt.subplots(2,3)

# simulate once for a static, once for a changing switch
for i,da in enumerate([0, 0.3]):
    dxdt = [1/epsilon*(f(ys(0), a+da*F(ys(1),Xc))*(ys(1)-ys(0))- g(ys(0))*ys(0)), kX-ys(1)*ys(0)]
    ggg = [sigma, 0] #noise
    ini = [0.01,0.01]
    sdesys = jitcsde(dxdt, ggg)
    sdesys.set_initial_value(ini,0.0)
    sdesys.set_integration_parameters(atol=1e-8,first_step=0.001, max_step=0.01,min_step=1e-13)
    timeseries = []
    tv = np.arange(sdesys.t, sdesys.t+200, 0.01)
    for time in tv:
        timeseries.append(sdesys.integrate(time) )

    timeseries = np.array(timeseries)
    Xv = timeseries[:,0]
    XTv = timeseries[:,1]

    axes[i,0].plot(XTv,Xv)
    axes[i,1].plot(tv, Xv, label='$X$')
    axes[i,1].plot(tv, XTv, label='$X_T$')

    #detect period using two thresholds
    # the function returns a list of tuples (time, up/down)
    ctt = csu.getcrossingtimes_twothresholds(tv, Xv, switch.folds[1][1], switch.folds[0][1])
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
    axes[i,1].axhline(switch.folds[0][1],color='k')
    axes[i,1].axhline(switch.folds[1][1],color='k')

# add labels etc
for ax in axes[:,0]: # phaseplanes
    ax.set_xlabel('$X_T$')
    ax.set_ylabel('$X$')
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)

# plot the switch curve for static switch
axes[0,0].plot(switch.xtv, switch.xv, 'k')


for ax in axes[:,1]: # time series
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()
    ax.set_xlim(0,20) # take a selection

axes[0,0].set_title('Static')
axes[1,0].set_title('Changing')



plt.show()
