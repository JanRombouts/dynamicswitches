This folder contains four Python modules written for the paper:

- changingswitches.py contains code to set up bistable response curves and implements a pseudo-arclength continuation algorithm to compute the response curve
- csutils.py contains various utility functions we used, such as period detection algorithms
- solveswitch.py contains the code for simulation of the spatial system with front propagation
- oneenzymessa.py contains the code for simulating the Gillespie algorithm version of the model

Note: these modules were not cleaned specifically before uploading. Clean code with explanation is provided in the files demo_*. These files were created to showcase a specific simulation/computation, usually corresponding to a specific figure in the paper.

- demo_drawresponsecurve.py implements code to draw the response curve of X as function of X_T for different values of a.
- demo_noisytransition.py implements a transition to high steady state for increasing X_T with noise (Langevin). It does this for a static and for a changing switch. It also shows the detection of the transition time.
- demo_noisytransition_ssa.py implements a transition to high steady state for increasing X_T using the Gillespie algorithm.
- demo_frontpropagation.py shows the simulation of the spatial system. With a constant value of a, the front travels at a single speed. With a heterogeneity, the front slows down and gets stuck. The output picture shows snapshots of the fronts over time, and the front position as function of time.
- demo_oscillation.py shows the oscillatory system without noise and with a time delay.
- demo_oscillation_langevin.py shows the oscillatory system, Langevin version. A histogram of periods obtained by detecting threshold crossings is shown as well.
- demo_oscillation_ssa.py shows the oscillatory system, fully stochastic version (Gillespie algorithm). A histogram of periods is also shown.
- demo_mitoticentry.py implements the simulation used for the biological example of mitotic entry with two cellular compartments.


The demo files can be run from the command line using python *filename*

Note: we use numpy, matplotlib, sympy and also the JITCODE, JITCDDE and JITCSDE packages for simulations. These need to be installed to run the demo files.
Documentation for the JITC*DE packages: https://jitcde-common.readthedocs.io/en/stable/
