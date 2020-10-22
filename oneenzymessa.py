# Module for the stochastic simulation algorithm

import numpy as np


class OneEnzymeSSA(object):
  def __init__(self, **params):
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']

        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.kX = params['kX']
        self.bdeg = params['bdeg'] # typically zero or one to turn degradation on or off

        self.af = params['a'] # base/mean value of a
        self.da = params['da']
        self.kappa = params['kappa']
        self.Xc = params['Xc'] # typically middle between folds

        self.epsilon = params['epsilon']
        self.omega = params['omega'] # system volume / scaling factor


  def f(self, x, a):
    return a + self.b*x**self.n / ((self.K*self.omega)**self.n + x**self.n)

  def g(self, x):
    return self.ap + self.bp*(self.Kp*self.omega)**self.m/((self.Kp*self.omega)**self.m + x**self.m)

  def F(self, xt):
      #note kappa is rescaled with omega too!
    return np.tanh(self.kappa/self.omega*(xt-self.Xc*self.omega))


  def getrate(self,i,X):
    # obtain the rate of reaction i, when in state X = [Xi, Xa]
    # order of reactions: activation, inactivation, production, degradation
    Xi, Xa = X
    a = self.af + self.da*self.F(Xi+Xa)

    if i==0:
      return Xi*self.f(Xa,a) / self.epsilon
    elif i==1:
      return Xa*self.g(Xa) / self.epsilon
    elif i==2:
      return self.kX*self.omega
    elif i==3:
      return Xa*Xi*self.bdeg/self.omega # can be zero
    elif i==4:
      return Xa*(Xa-1)*self.bdeg/self.omega

  def getstoich(self, i):
      # stoichometry (change in molecule numbers) of reaction i
      if i==0:
          return [-1,1] # activation
      elif i==1:
          return [1, -1] # inactivation
      elif i==2:
          return [1, 0] # production
      elif i==3:
          return [-1, 0] # degradation of Xi
      elif i==4:
          return [0, -1] # degradation of Xa

  def simulate(self, T, dt, init=[0,0]):
      # simulates until time T is crossed
      rng = np.random.default_rng() # random number generator

      t_eval = np.linspace(0, T, int(T/dt)+1) # times at which to save sample

      time = 0.

      Xi, Xa = init
      XX = [[Xi, Xa]]

      tpos=1 # next position in t_eval
      while time < T:
          # sample time of next reaction
          R = sum([self.getrate(i,[Xi,Xa]) for i in range(5)])
          tau = -np.log(rng.random())/R # time to next reaction
          # determine which reaction takes place
          r = rng.random()
          s = self.getrate(0,[Xi,Xa])
          i=0
          while s/R < r:
              s += self.getrate(i+1, [Xi,Xa])
              i += 1
          # the i th reaction is chosen

          # update state
          deltaX = self.getstoich(i)
          Xi += deltaX[0]
          Xa += deltaX[1]

          time += tau
          while tpos<len(t_eval) and time > t_eval[tpos] : # while is needed since it is possible that if tau is large, multiple sample times are passed
            XX.append([Xi, Xa])
            tpos+=1

      XX = np.array(XX)
      self.tt = t_eval
      self.Xi = XX[:,0]
      self.Xa = XX[:,1]
