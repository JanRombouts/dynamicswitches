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
from jitcode import jitcode, y, t


#standard Yang and Ferrell parameter set but without the degradation (adeg and bdeg = 0)
# ks=1.5
p = {'ks': 1.5, 'adeg': 0.0, 'bdeg':0.0 , 'ndeg': 17, 'Kdeg': 32., \
                 'acdc25': 0.16, 'bcdc25': 0.8, 'ncdc25': 11, 'Kcdc25': 35., \
                 'awee1': 0.08, 'bwee1': 0.4, 'nwee1': 3.5, 'Kwee1':30. }

# import export rates
pr = {'kncyc': .05, 'kccyc': .05, 'kncdc': 0.05, 'kccdc': .05}

#cdc25 and wee1 activity depend on cdk1, but also on their total
def cdc25(cdk,cdc25T=1):
    return cdc25T*(p['acdc25'] + p['bcdc25']*cdk**p['ncdc25']/(p['Kcdc25']**p['ncdc25'] + cdk**p['ncdc25']))
def wee1(cdk,wee1T=1):
    return wee1T*(p['awee1'] + p['bwee1']*p['Kwee1']**p['nwee1']/(p['Kwee1']**p['nwee1'] + cdk**p['nwee1']))

#solve
#order: cyc, cdk, cdc
#nucleus cytoplasm
cycn = y(0)
cycc = y(1)

cdkn = y(2)
cdkc = y(3)

cdcn = y(4)
cdcc = y(5)

#import rates depend on Cdk1 levels
impcdc = .1+cdkc/30 #import factor for cdc25
impcyc = 1 +cdkn/60 #import factor for cyclin

# derivatives
dcycn = pr['kncyc']*impcyc*cycc - pr['kccyc']*cycn
dcycc = -pr['kncyc']*impcyc*cycc + pr['kccyc']*cycn + p['ks']

dcdkn = pr['kncyc']*impcyc*cdkc - pr['kccyc']*cdkn + cdc25(cdkn,cdcn)*(cycn-cdkn)\
   - wee1(cdkn, 1.3)*cdkn
dcdkc = -pr['kncyc']*impcyc*cdkc + pr['kccyc']*cdkn + p['ks'] \
     + cdc25(cdkc,cdcc)*(cycc-cdkc) - wee1(cdkc,1)*cdkc


dcdcn = pr['kncdc']*cdcc*impcdc - pr['kccdc']*cdcn
dcdcc = -pr['kncdc']*cdcc*impcdc + pr['kccdc']*cdcn

# set up the ODE solver
dxdt = [dcycn, dcycc, dcdkn, dcdkc, dcdcn, dcdcc]
y0 = [0.,0.,0.,0.,1.,2.]

odesys = jitcode(dxdt)

odesys.set_integrator("dopri5")
odesys.set_initial_value(y0,0.0)

data = []

#integrate
tv = np.linspace(0, 150, 1000)
for time in tv:
    data.append(odesys.integrate(time))

data = np.array(data)

cycnv= data[:,0]
cyccv = data[:,1]

cdknv = data[:,2]
cdkcv = data[:,3]

cdcnv = data[:,4]
cdccv = data[:,5]

# Plotting

#timeseries
fig, axes = plt.subplots(1,2)

# plot time series in nucleus (left) and cytoplasm(right)
# Cdc25 times 20 just for visuals
for x,lb in zip([cycnv,cdknv,cdcnv*20], ['Cyc', 'Cdk', 'Cdc25 (20x)']):
    l, = axes[0].plot(tv, x, label=lb)
for x,lb in zip([cyccv,cdkcv,cdccv*20], ['Cyc', 'Cdk', 'Cdc25 (20x)']):
    l, = axes[1].plot(tv, x, label=lb)


for ax in axes:
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.legend()

axes[0].set_title('Nucleus')
axes[1].set_title('Cytoplasm')
# next: phaseplane for nucleus and cytoplasm
fig, axes = plt.subplots(1,2)

for ax in axes:
    ax.set_xlabel('Cyclin B')
    ax.set_ylabel('Active Cdk1')

    ax.set_xlim(0,100)
    ax.set_ylim(0,100)

axes[0].set_title('Nucleus')
axes[1].set_title('Cytoplasm')

# set up switch

switch= cs.ResponseCdk1(**p)
switch.setcontpars(0.15,2000)
switch.setstart(0,0,[0.1,0.1])

axes[0].plot(cycnv,cdknv)
axes[1].plot(cyccv,cdkcv)

#snapshots of the bistable switch
n = 5
totsteps = len(tv)

for i in range(0, totsteps, int(totsteps//n)):
    switch.acdc25 = p['acdc25']*cdcnv[i]
    switch.bcdc25 = p['bcdc25']*cdcnv[i]
    switch.awee1 = p['awee1']*1.3 #values of wee1 are higher in nucleus
    switch.bwee1 = p['bwee1']*1.3
    switch.compute_responsecurve()

    axes[0].plot(switch.cycv, switch.cdkv, alpha=i/totsteps, color='k')

    switch.acdc25 = p['acdc25']*cdccv[i]
    switch.bcdc25 = p['bcdc25']*cdccv[i]
    switch.awee1 = p['awee1']*1 #values of wee1 in cytoplasm
    switch.bwee1 = p['bwee1']*1
    switch.compute_responsecurve()

    axes[1].plot(switch.cycv, switch.cdkv, alpha=i/totsteps, color='k')

plt.show()
