
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from scipy.fftpack import fft,ifft,fftshift,ifftshift
from scipy.optimize import leastsq

class Problem(object):
    """This class contains the parameters
    """
    def __init__(self, **params):
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']

        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.DX = params['DX'] #diffusion constant
        self.DY = params['DY'] #diffusion constant inactive form

        #af and XTf are functions of space and time
        self.af = params['af']
        self.kXTf = params['kXTf'] #source of total enzyme

        if 'epsilon' in params.keys():
            self.epsilon = params['epsilon']
        else:
            self.epsilon = 1./10

    def f(self,a, X):
        return a + self.b * X**self.n/(self.K**self.n + X**self.n)

    def g(self,X):
        return self.ap + self.bp * self.Kp ** self.m / (self.Kp**self.m + X**self.m)

    def tostring(self):
        #gives String representation containing the parameters
        s = ' '.join(['{:.6f}'.format(pp) for pp in [self.ap,self.b, self.bp, self.K, self.Kp, self.n, self.m, self.epsilon, self.DX, self.DY]])
        return s
    def tostring_header(self):
        s = 'ap b bp K Kp n m epsilon DX DY'
        return s


class Profile(object):
    """class contains x, X, Y values for a profile"""
    def __init__(self, *args):
        if len(args)==1:
            #all given in one array
            d = args[0]
            if d.shape[0]==2:
                d=d.T
            self.xv = d[:,0]
            self.Xv = d[:,1]
            self.Yv = d[:,2]
        else:
            #three separate one dimensional arrays
            self.xv = args[0]
            self.Xv = args[1]
            self.Yv = args[2]

class Solver(object):
        @staticmethod
        def laplacian(u, dx, bc='periodic'):
          """Returns Laplacian with given BC"""
          l = 1./dx**2*(np.roll(u, -1) - 2*u + np.roll(u,1))
          if bc == 'neumann':
              l[0]=2*(u[1]-u[0])/dx**2
              l[-1]=2*(u[-2]-u[-1])/dx**2
          return l

        @staticmethod
        def firstder(u, dx, bc='periodic'):
            """first spatial derivative"""
            l = 1./2./dx*(np.roll(u,-1) - np.roll(u,1))
            if bc=='neumann':
                l[0]=0
                l[-1]=0
            return l
        @staticmethod
        def H(x):
          """Heaviside function"""
          return 0.5*(np.sign(x)+1)

        def __init__(self, problem, initprofilefunction, **numpars):
            """initprofilefunction is a function which takes one argument x and returns a tuple (u,v)"""

            self.problem = problem
            self.T = numpars['T']
            self.L = numpars['L']
            self.dt = numpars['dt']
            self.N = numpars['N']
            self.dx = 1.*self.L/self.N
            self.outputstep=numpars['outputstep']
            self.bc=numpars['bc']
            self.method=numpars['method']

            if 'nu' in numpars.keys():
                self.nu=numpars['nu']
            else: self.nu=0.

            acceptedmethods = ['forwardeuler', 'splitting'] #, 'backwardeuler_picard', 'pseudospectral', 'splitting']
            if self.method not in acceptedmethods:
                print('method not in accepted methods, taking forward euler')
                self.method = 'forwardeuler'

            #setup mesh
            self.x = np.linspace(0,self.L,self.N)
            #set initial profile
            X = np.zeros_like(self.x)
            Y = np.zeros_like(self.x)

            #init function provides X,Y values for each space unit
            for i,xx in enumerate(self.x):
                xxx,yyy=initprofilefunction(xx)
                X[i]=xxx
                Y[i]=yyy

            self.initprofile = Profile(self.x,X,Y)


            #if we are using bw euler, set the matrices for u and v

            if self.method=='backwardeuler_picard':
                self.theta=1
            elif self.method=='splitting':
                self.theta=0.5
            else:
                self.theta=0.

            if self.method != 'forwardeuler':
                # setup matrices for solver
                self.set_diffusion(self.problem.DX,self.problem.DY)

        def set_profile(self, prof):
            """sets the initial profile, but checks if it corresponds in length"""
            if len(prof.xv) != self.N:
                print("the length of the new profile does not correspond to the length of self.x ")
                print("profile has not been changed.")
            else:
                self.initprofile=prof

        def set_diffusion(self, DX,DY):
            """Function to set the diffusion constants. We cannot just change the problem parameter, since
            FX, AX depend on this.  """
            self.problem.DX = DX
            self.problem.DY = DY

            FX = self.dt/self.dx**2*self.problem.DX
            FY = self.dt/self.dx**2*self.problem.DY

            AX = scipy.sparse.diags(diagonals=
                [[1+2*FX*self.theta]*self.N, [-FX*self.theta]*(self.N-1),[-FX*self.theta]*(self.N-1)],
                offsets=[0,1,-1],format='csr')

            AY = scipy.sparse.diags(diagonals=
                [[1+2*FY*self.theta]*self.N, [-FY*self.theta]*(self.N-1),[-FY*self.theta]*(self.N-1)],
                offsets=[0,1,-1],format='csr')


            if self.bc == 'periodic':
                AX[0,-1]=-FX*self.theta
                AX[-1,0]=-FX*self.theta
                AY[0,-1]=-FY*self.theta
                AY[-1,0]=-FY*self.theta

            elif self.bc == 'neumann':
                AX[0,1]=-2*FX*self.theta
                AX[-1,-2]=-2*FX*self.theta
                AY[0,1]=-2*FY*self.theta
                AY[-1,-2]=-2*FY*self.theta


            self.AX=AX
            self.FX=FX
            self.AY=AY
            self.FY=FY

        def solve(self):
            """Solve the system by timestepping"""
            #steps, computed from T and dt
            self.steps = int(self.T/self.dt)

            #setup the matrices
            if self.steps % self.outputstep ==0:
                dimX = int(self.steps/self.outputstep)
            else:
                dimX = int(self.steps/self.outputstep+1)
            self.XX = np.zeros((dimX, self.N))
            self.YY = np.zeros((dimX, self.N))


            #tt contains the time values at which we save
            self.tt = np.linspace(0,self.T,dimX)
            #set initial
            self.XX[0,:] = self.initprofile.Xv
            self.YY[0,:] = self.initprofile.Yv


            #starting values
            X = self.initprofile.Xv
            Y = self.initprofile.Yv

            if self.method=='forwardeuler':
                for i in range(1,self.steps):
                    #compute a and XT (can be space and time dependent)
                    a = self.problem.af(self.x, i*self.dt)

                    kXT = self.problem.kXTf(self.x,i*self.dt)

                    Xnew = X + self.dt*(self.problem.DX*self.laplacian(X, self.dx, bc=self.bc) \
                        + 1./self.problem.epsilon*(self.problem.f(a,X)*Y - self.problem.g(X)*X))

                    Ynew = Y + self.dt*(self.problem.DY*self.laplacian(Y, self.dx, bc=self.bc) \
                        + 1./self.problem.epsilon*(-self.problem.f(a,X)*Y + self.problem.g(X)*X) + kXT)

                    Xnew += self.nu*np.random.normal(0,1,X.shape)
                    Ynew += self.nu*np.random.normal(0,1,Y.shape)

                    if i % self.outputstep == 0:
                        self.XX[int(i/self.outputstep),:] = X
                        self.YY[int(i/self.outputstep),:] = Y
                    X=Xnew
                    Y=Ynew

            elif self.method=='splitting':
                #use an operator-splitting method
                for i in range(1,self.steps):
                    #first reaction part over half a timestep using a second order method (eg. Heun)
                    a = self.problem.af(self.x, i*self.dt)
                    kXT = self.problem.kXTf(self.x, i*self.dt)

                    Xs = X + self.dt/2*1./self.problem.epsilon*(self.problem.f(a,X)*Y - self.problem.g(X)*X)
                    Ys = Y + self.dt/2*(1./self.problem.epsilon*(-self.problem.f(a,X)*Y + self.problem.g(X)*X)+kXT)


                    ass = self.problem.af(self.x, (i+0.5)*self.dt)
                    kXTss = self.problem.kXTf(self.x, (i+0.5)*self.dt)

                    Xss=X+0.5*self.dt/2*1./self.problem.epsilon*(self.problem.f(a,X)*Y - self.problem.g(X)*X \
                        + self.problem.f(ass,Xs)*Ys - self.problem.g(Xs)*Xs)
                    Yss=Y+0.5*self.dt/2*(1./self.problem.epsilon*(-self.problem.f(a,X)*Y + self.problem.g(X)*X \
                        - self.problem.f(ass,Xs)*Ys + self.problem.g(Xs)*Xs)+kXT+kXTss)

                    #now diffusion step using theta-rule over one whole timestep
                    #need dx here to compensate for laplacian and Fu
                    rhsX=self.FX*(1-self.theta)*self.dx**2*self.laplacian(Xss, self.dx, self.bc)+Xss
                    Xsss=scipy.sparse.linalg.spsolve(self.AX,rhsX)

                    rhsY=self.FY*(1-self.theta)*self.dx**2*self.laplacian(Yss, self.dx, self.bc)+Yss
                    Ysss=scipy.sparse.linalg.spsolve(self.AY,rhsY)


                    #finally one more reaction step
                    Xs=Xsss + self.dt/2*1./self.problem.epsilon*(self.problem.f(ass,Xsss)*Ysss - self.problem.g(Xsss)*Xsss)
                    Ys=Ysss + self.dt/2*(1./self.problem.epsilon*(-self.problem.f(ass,Xsss)*Ysss + self.problem.g(Xsss)*Xsss)+kXTss)

                    an = self.problem.af(self.x, (i+1)*self.dt)
                    kXTn = self.problem.kXTf(self.x, (i+1)*self.dt)

                    Xnew=Xsss+0.5*self.dt/2*1./self.problem.epsilon*(self.problem.f(ass,Xsss)*Ysss - self.problem.g(Xsss)*Xsss \
                    + self.problem.f(an,Xs)*Ys - self.problem.g(Xs)*Xs)
                    Ynew=Ysss+0.5*self.dt/2*(1./self.problem.epsilon*(-self.problem.f(ass,Xsss)*Ysss + self.problem.g(Xsss)*Xsss \
                    - self.problem.f(an,Xs)*Ys + self.problem.g(Xs)*Xs) + kXTss+kXTn)


                    if i % self.outputstep == 0:
                        self.XX[int(i/self.outputstep),:] = X
                        self.YY[int(i/self.outputstep),:] = Y
                    X=Xnew
                    Y=Ynew


        def paramstostring(self):
            # writes out all the numerical parameters in a string
            s = '{:.6f} {:d} {:.6f} {:.6f} {:d} {:s} {:s}'.format(self.L, self.N, self.T, self.dt, self.outputstep, self.bc, self.method)
            s = s + ' {:.6f} {:.6f}'.format(self.XX[0,0], self.YY[0,0]) #initial values

        def paramstostring_header(self):
            # description of the parameters that are output by paramstostring
            s = 'L N T dt outputstep bc method initX initY'
            return s


        def getfrontposition(self, thr=None, flip=False):
            """ Assuming the high activity is on the right, unless flip is True.
            For every timestep, we calculate the position where the profile goes
            from below thr on the left, to above thr on the right"""
            if thr == None: # take the average of the low and high, after some timesteps
                ts = 20
                thr = 0.5*(self.XX[ts,0] + self.XX[ts,-1])
            self.frontposthreshold = thr # keep for later

            pos = np.zeros_like(self.tt)
            for step in range(self.XX.shape[0]):
                currentprofile = self.XX[step,:]
                if flip:
                    currentprofile = currentprofile[-1::-1]
                if currentprofile[0] > thr or currentprofile[-1] < thr:
                    pos[step]=-1
                else:
                    i=0
                    while currentprofile[i] < thr and i < len(currentprofile)-1:
                        i += 1
                    pos[step] = 0.5*(self.x[i-1] + self.x[i]) # approximate x value of crossing point
                    if flip:
                        pos[step] = self.L - pos[step]
            self.frontposition = pos
