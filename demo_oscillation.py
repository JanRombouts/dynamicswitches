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

for i,aa in enumerate(np.linspace(lowa,higha,n)):
    switch.a=aa
    switch.compute_responsecurve()
    axes[1,0].plot(switch.xtv, switch.xv, color='k', alpha=i/n)

for ax in axes[:,1]: # time series
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()

axes[0,0].set_title('Static')
axes[1,0].set_title('Changing')

plt.show()
