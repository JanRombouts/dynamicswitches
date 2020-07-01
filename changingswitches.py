#module that contains utility functions bistable switches

import numpy as np
from scipy.optimize import fsolve

class ResponseOneEnzyme(object):

    def __init__(self, **params):
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']

        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.Xt = params['Xt']


    def setcontpars(self, h, totit):
        self.h=h
        self.totit=totit

    def setstart(self,a,x,v):
        #set starting point for continuation
        self.starta=a
        self.startx=x
        self.startv=v

    def predict(self,a,x,v,h):
        #v is the previous tangent vector
        #h is the step size
        #a is the kinase strength
        #x is enzyme activity

        #define local vars for lighter code
        b,K,n = self.b, self.K, self.n
        ap, bp, Kp, m = self.ap, self.bp, self.Kp, self.m
        Xt = self.Xt

        #compute hill functions
        f=a+b*x**n/(K**n+x**n)
        g=ap+bp*Kp**m/(Kp**m+x**m)

        #derivatives of hill functions
        fa = 1.
        fx = b*n*x**(n-1)*K**n/(K**n+x**n)**2
        gx=-bp*Kp**m*m*x**(m-1)/(Kp**m+x**m)**2

        #jacobian
        Fa=fa*(Xt-x)
        Fx=fx*(Xt-x) - f-gx*x-g

        #steps for a and x
        a_s = -Fx/(Fa*v[1] - Fx*v[0])
        x_s = Fa/(Fa*v[1] - Fx*v[0])

        #normalize
        r = np.sqrt(a_s**2 + x_s**2)
        a_s /= r
        x_s /= r

        return a+h*a_s, x+h*x_s, a_s, x_s

    def correct(self,a0,x0, a_s, x_s):
        #solve to lie on curve with additional condition orthogonal to previous step
        #define local vars for lighter code
        b,K,n = self.b, self.K, self.n
        ap, bp, Kp, m = self.ap, self.bp, self.Kp, self.m
        Xt = self.Xt

        def F(a,x):
            return (a+b*x**n/(K**n+x**n))*(Xt-x) - (ap+bp*Kp**m/(Kp**m+x**m))*x

        def tosolve(X):
            a=X[0]
            x=X[1]
            return [F(a,x), (a0-a)*a_s + (x0-x)*x_s]
        return fsolve(tosolve, [a0,x0])

    def compute_responsecurve(self):
        #sets a and x values as array
        #also the v values
        #and sets the fold points

        av = [self.starta]
        xv = [self.startx]
        vv = [ [self.startv[0], self.startv[1]] ]

        h=self.h
        totit = self.totit

        folds = []
        for i in range(totit):
            a0,x0,v0,v1 = self.predict(av[-1], xv[-1], vv[-1], h)
            a1,x1 = self.correct(a0,x0,v0,v1)
            av.append(a1)
            xv.append(x1)
            vv.append([v0,v1])

            #if the first component of v changes sign, we have a fold
            if vv[-1][0]*vv[-2][0] < 0:
                folds.append((a1,x1))

        self.av = np.array(av)
        self.xv = np.array(xv)
        self.vv = vv
        self.folds = folds

class ResponseOneEnzymeXt(object):
    """ Response to changing Xt, with fixed a"""
    def __init__(self, **params):
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']

        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.a = params['a']


    def setcontpars(self, h, totit):
        self.h=h
        self.totit=totit

    def setstart(self,xt,x,v):
        #set starting point for continuation
        self.startxt=xt
        self.startx=x
        self.startv=v

    def predict(self,xt,x,v,h):
        #v is the previous tangent vector
        #h is the step size
        #a is the total x
        #x is enzyme activity

        #define local vars for lighter code
        b,K,n = self.b, self.K, self.n
        ap, bp, Kp, m = self.ap, self.bp, self.Kp, self.m
        a = self.a

        #compute hill functions
        f=a+b*x**n/(K**n+x**n)
        g=ap+bp*Kp**m/(Kp**m+x**m)

        #derivatives of hill functions
        fa = 1.
        fx = b*n*x**(n-1)*K**n/(K**n+x**n)**2
        gx=-bp*Kp**m*m*x**(m-1)/(Kp**m+x**m)**2

        #jacobian
        Fxt=f
        Fx=fx*(xt-x) - f-gx*x-g

        #steps for a and x
        xt_s = -Fx/(Fxt*v[1] - Fx*v[0])
        x_s = Fxt/(Fxt*v[1] - Fx*v[0])

        #normalize
        r = np.sqrt(xt_s**2 + x_s**2)
        xt_s /= r
        x_s /= r

        return xt+h*xt_s, x+h*x_s, xt_s, x_s

    def correct(self,xt0,x0, xt_s, x_s):
        #solve to lie on curve with additional condition orthogonal to previous step
        #define local vars for lighter code
        b,K,n = self.b, self.K, self.n
        ap, bp, Kp, m = self.ap, self.bp, self.Kp, self.m
        a = self.a

        def F(xt,x):
            return (a+b*x**n/(K**n+x**n))*(xt-x) - (ap+bp*Kp**m/(Kp**m+x**m))*x

        def tosolve(X):
            xt=X[0]
            x=X[1]
            return [F(xt,x), (xt0-xt)*xt_s + (x0-x)*x_s]
        return fsolve(tosolve, [xt0,x0])

    def compute_responsecurve(self):
        #sets a and x values as array
        #also the v values
        #and sets the fold points

        xtv = [self.startxt]
        xv = [self.startx]
        vv = [ [self.startv[0], self.startv[1]] ]

        h=self.h
        totit = self.totit

        folds = []
        for i in range(totit):
            xt0,x0,v0,v1 = self.predict(xtv[-1], xv[-1], vv[-1], h)
            xt1,x1 = self.correct(xt0,x0,v0,v1)
            xtv.append(xt1)
            xv.append(x1)
            vv.append([v0,v1])

            #if the first component of v changes sign, we have a fold
            if vv[-1][0]*vv[-2][0] < 0:
                folds.append((xt1,x1))

        self.xtv = np.array(xtv)
        self.xv = np.array(xv)
        self.vv = vv
        self.folds = folds

class ResponseOneEnzymeXt_withdeg(object):
    """ Response to changing Xt, with fixed a. Degradation of X included"""
    def __init__(self, **params):
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']

        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.a = params['a']
        self.epsilon = params['epsilon']

    def setcontpars(self, h, totit):
        self.h=h
        self.totit=totit

    def setstart(self,xt,x,v):
        #set starting point for continuation
        self.startxt=xt
        self.startx=x
        self.startv=v

    def predict(self,xt,x,v,h):
        #v is the previous tangent vector
        #h is the step size
        #a is the total x
        #x is enzyme activity

        #define local vars for lighter code
        b,K,n = self.b, self.K, self.n
        ap, bp, Kp, m = self.ap, self.bp, self.Kp, self.m
        a = self.a
        eps = self.epsilon
        #compute hill functions
        f=a+b*x**n/(K**n+x**n)
        g=ap+bp*Kp**m/(Kp**m+x**m)

        #derivatives of hill functions
        fa = 1.
        fx = b*n*x**(n-1)*K**n/(K**n+x**n)**2
        gx=-bp*Kp**m*m*x**(m-1)/(Kp**m+x**m)**2

        #jacobian
        #assume h(x) = x degradation function
        Fxt=f*eps**(-1)
        Fx=eps**(-1)*(fx*(xt-x) - f-gx*x-g) - 2*x

        #steps for a and x
        xt_s = -Fx/(Fxt*v[1] - Fx*v[0])
        x_s = Fxt/(Fxt*v[1] - Fx*v[0])

        #normalize
        r = np.sqrt(xt_s**2 + x_s**2)
        xt_s /= r
        x_s /= r

        return xt+h*xt_s, x+h*x_s, xt_s, x_s

    def correct(self,xt0,x0, xt_s, x_s):
        #solve to lie on curve with additional condition orthogonal to previous step
        #define local vars for lighter code
        b,K,n = self.b, self.K, self.n
        ap, bp, Kp, m = self.ap, self.bp, self.Kp, self.m
        a = self.a
        eps = self.epsilon

        def F(xt,x):
            return eps**(-1)*((a+b*x**n/(K**n+x**n))*(xt-x) - (ap+bp*Kp**m/(Kp**m+x**m))*x) - x**2

        def tosolve(X):
            xt=X[0]
            x=X[1]
            return [F(xt,x), (xt0-xt)*xt_s + (x0-x)*x_s]
        return fsolve(tosolve, [xt0,x0])

    def compute_responsecurve(self):
        #sets a and x values as array
        #also the v values
        #and sets the fold points

        xtv = [self.startxt]
        xv = [self.startx]
        vv = [ [self.startv[0], self.startv[1]] ]

        h=self.h
        totit = self.totit

        folds = []
        for i in range(totit):
            xt0,x0,v0,v1 = self.predict(xtv[-1], xv[-1], vv[-1], h)
            xt1,x1 = self.correct(xt0,x0,v0,v1)
            xtv.append(xt1)
            xv.append(x1)
            vv.append([v0,v1])

            #if the first component of v changes sign, we have a fold
            if vv[-1][0]*vv[-2][0] < 0:
                folds.append((xt1,x1))

        self.xtv = np.array(xtv)
        self.xv = np.array(xv)
        self.vv = vv
        self.folds = folds


class ResponseCdk1(object):

    def __init__(self, **params):
        #parameters
        #first the degradation (APC/C) parameters
        self.ks = params['ks']
        self.adeg = params['adeg']
        self.bdeg = params['bdeg']
        self.ndeg = params['ndeg']
        self.Kdeg = params['Kdeg']

        #cdc25 loop
        self.acdc25 = params['acdc25']
        self.bcdc25 = params['bcdc25']
        self.ncdc25 = params['ncdc25']
        self.Kcdc25 = params['Kcdc25']

        #wee1 loop
        self.awee1 = params['awee1']
        self.bwee1 = params['bwee1']
        self.nwee1 = params['nwee1']
        self.Kwee1 = params['Kwee1']


    def setcontpars(self, h, totit):
        self.h=h
        self.totit=totit

    def setstart(self,cyc,cdk,v):
        #set starting point for continuation
        self.startcyc=cyc
        self.startcdk=cdk
        self.startv=v

    def predict(self,cyc,cdk,v,h):
        #v is the previous tangent vector
        #h is the step size
        #cyc is the Cyclin B concentration
        #cdk is Cdk1 activity

        #define local vars for lighter code
        adeg, bdeg, ndeg, Kdeg = self.adeg,self.bdeg,self.ndeg,self.Kdeg
        acdc25, bcdc25, ncdc25, Kcdc25 = self.acdc25, self.bcdc25, self.ncdc25, self.Kcdc25
        awee1, bwee1, nwee1, Kwee1 = self.awee1, self.bwee1, self.nwee1, self.Kwee1

        #compute cdc25, wee1 and apc/c as function of cdk1
        cdc25 = acdc25 + bcdc25*cdk**ncdc25/(Kcdc25**ncdc25 + cdk**ncdc25)
        wee1 = awee1 + bwee1*Kwee1**nwee1/(Kwee1**nwee1 + cdk**nwee1)
        apc = adeg + bdeg*cdk**ndeg/(Kdeg**ndeg+cdk**ndeg)

        #derivatives of hill functions for cdc25,wee1,apc
        cdc25x = bcdc25*ncdc25*Kcdc25**ncdc25*cdk**(ncdc25-1)/(Kcdc25**ncdc25 + cdk**ncdc25)**2
        wee1x = -nwee1*bwee1*Kwee1**nwee1*cdk**(nwee1-1)/(Kwee1**nwee1 + cdk**nwee1)**2
        apcx = bdeg*ndeg*Kdeg**ndeg*cdk**(ndeg-1)/(Kdeg**ndeg + cdk**ndeg)**2

        #jacobian
        Fx = cdc25x*(cyc-cdk) - cdc25 - wee1x*cdk-wee1-apcx*cdk - apc
        Fy = cdc25

        #steps for a and x. v is solution of 2x2 system
        cyc_s = -Fx/(Fy*v[1] - Fx*v[0])
        cdk_s = Fy/(Fy*v[1] - Fx*v[0])

        #normalize
        r = np.sqrt(cyc_s**2 + cdk_s**2)
        cyc_s /= r
        cdk_s /= r

        return cyc+h*cyc_s, cdk+h*cdk_s, cyc_s, cdk_s

    def correct(self,cyc0,cdk0, cyc_s, cdk_s):
        #solve to lie on curve with additional condition orthogonal to previous step
        #define local vars for lighter code
        ks = self.ks
        adeg, bdeg, ndeg, Kdeg = self.adeg,self.bdeg,self.ndeg,self.Kdeg
        acdc25, bcdc25, ncdc25, Kcdc25 = self.acdc25, self.bcdc25, self.ncdc25, self.Kcdc25
        awee1, bwee1, nwee1, Kwee1 = self.awee1, self.bwee1, self.nwee1, self.Kwee1

        def F(cyc,cdk):
            cdc25 = acdc25 + bcdc25*cdk**ncdc25/(Kcdc25**ncdc25 + cdk**ncdc25)
            wee1 = awee1 + bwee1*Kwee1**nwee1/(Kwee1**nwee1 + cdk**nwee1)
            apc = adeg + bdeg*cdk**ndeg/(Kdeg**ndeg+cdk**ndeg)
            return ks + cdc25*(cyc-cdk) - wee1*cdk - apc*cdk

        def tosolve(X):
            cyc=X[0]
            cdk=X[1]
            return [F(cyc,cdk), (cyc0-cyc)*cyc_s + (cdk0-cdk)*cdk_s]
        return fsolve(tosolve, [cyc0,cdk0])

    def compute_responsecurve(self):
        #sets cyc and cdk values as array
        #also the v values
        #and sets the fold points

        cycv = [self.startcyc]
        cdkv = [self.startcdk]
        vv = [ [self.startv[0], self.startv[1]] ]

        h=self.h
        totit = self.totit

        folds = []
        for i in range(totit):
            cyc0,cdk0,v0,v1 = self.predict(cycv[-1], cdkv[-1], vv[-1], h)
            cyc1,cdk1 = self.correct(cyc0,cdk0,v0,v1)
            cycv.append(cyc1)
            cdkv.append(cdk1)
            vv.append([v0,v1])

            #if the first component of v changes sign, we have a fold
            if vv[-1][0]*vv[-2][0] < 0:
                folds.append((cyc1,cdk1))

        self.cycv = np.array(cycv)
        self.cdkv = np.array(cdkv)
        self.vv = vv
        self.folds = folds

class Nullcline(object):
    # draws response of Cdk1 tot cyclin, but with a dependence of cc25 on cyclin too
    def __init__(self, **params):
        #parameters
        #first the degradation (APC/C) parameters
        self.ks = params['ks']
        self.adeg = params['adeg']
        self.bdeg = params['bdeg']
        self.ndeg = params['ndeg']
        self.Kdeg = params['Kdeg']

        #cdc25 loop
        self.acdc25 = params['acdc25']
        self.bcdc25 = params['bcdc25']
        self.ncdc25 = params['ncdc25']
        self.Kcdc25 = params['Kcdc25']

        #cdc25 is modified by cyclin activity
        self.cycfactor = params['cycfactor'] #this is a function

        #wee1 loop
        self.awee1 = params['awee1']
        self.bwee1 = params['bwee1']
        self.nwee1 = params['nwee1']
        self.Kwee1 = params['Kwee1']


    def setcontpars(self, h, totit):
        self.h=h
        self.totit=totit

    def setstart(self,cyc,cdk,v):
        #set starting point for continuation
        self.startcyc=cyc
        self.startcdk=cdk
        self.startv=v

    def predict(self,cyc,cdk,v,h):
        #v is the previous tangent vector
        #h is the step size
        #cyc is the Cyclin B concentration
        #cdk is Cdk1 activity

        #define local vars for lighter code
        adeg, bdeg, ndeg, Kdeg = self.adeg,self.bdeg,self.ndeg,self.Kdeg
        acdc25, bcdc25, ncdc25, Kcdc25 = self.acdc25, self.bcdc25, self.ncdc25, self.Kcdc25
        awee1, bwee1, nwee1, Kwee1 = self.awee1, self.bwee1, self.nwee1, self.Kwee1

        cf = self.cycfactor(cyc)

        #compute cdc25, wee1 and apc/c as function of cdk1
        cdc25 = (acdc25 + bcdc25*cdk**ncdc25/(Kcdc25**ncdc25 + cdk**ncdc25))*cf
        wee1 = awee1 + bwee1*Kwee1**nwee1/(Kwee1**nwee1 + cdk**nwee1)
        apc = adeg + bdeg*cdk**ndeg/(Kdeg**ndeg+cdk**ndeg)

        #derivatives of hill functions for cdc25,wee1,apc
        cdc25x = (bcdc25*ncdc25*Kcdc25**ncdc25*cdk**(ncdc25-1)/(Kcdc25**ncdc25 + cdk**ncdc25)**2)*cf
        wee1x = -nwee1*bwee1*Kwee1**nwee1*cdk**(nwee1-1)/(Kwee1**nwee1 + cdk**nwee1)**2
        apcx = bdeg*ndeg*Kdeg**ndeg*cdk**(ndeg-1)/(Kdeg**ndeg + cdk**ndeg)**2

        hcyc = 0.00001
        cfy = (self.cycfactor(cyc+hcyc)-self.cycfactor(cyc-hcyc))/2/hcyc # numerical derivatives

        #jacobian
        Fx = cdc25x*(cyc-cdk) - cdc25 - wee1x*cdk-wee1-apcx*cdk - apc
        Fy = cdc25 + (cyc-cdk)*cdc25/cf*cfy

        #steps for a and x. v is solution of 2x2 system
        cyc_s = -Fx/(Fy*v[1] - Fx*v[0])
        cdk_s = Fy/(Fy*v[1] - Fx*v[0])

        #normalize
        r = np.sqrt(cyc_s**2 + cdk_s**2)
        cyc_s /= r
        cdk_s /= r

        return cyc+h*cyc_s, cdk+h*cdk_s, cyc_s, cdk_s

    def correct(self,cyc0,cdk0, cyc_s, cdk_s):
        #solve to lie on curve with additional condition orthogonal to previous step
        #define local vars for lighter code
        ks = self.ks
        adeg, bdeg, ndeg, Kdeg = self.adeg,self.bdeg,self.ndeg,self.Kdeg
        acdc25, bcdc25, ncdc25, Kcdc25 = self.acdc25, self.bcdc25, self.ncdc25, self.Kcdc25
        awee1, bwee1, nwee1, Kwee1 = self.awee1, self.bwee1, self.nwee1, self.Kwee1

        def F(cyc,cdk):
            cdc25 = (acdc25 + bcdc25*cdk**ncdc25/(Kcdc25**ncdc25 + cdk**ncdc25))*self.cycfactor(cyc)
            wee1 = awee1 + bwee1*Kwee1**nwee1/(Kwee1**nwee1 + cdk**nwee1)
            apc = adeg + bdeg*cdk**ndeg/(Kdeg**ndeg+cdk**ndeg)
            return ks + cdc25*(cyc-cdk) - wee1*cdk - apc*cdk

        def tosolve(X):
            cyc=X[0]
            cdk=X[1]
            return [F(cyc,cdk), (cyc0-cyc)*cyc_s + (cdk0-cdk)*cdk_s]
        return fsolve(tosolve, [cyc0,cdk0])

    def compute_responsecurve(self):
        #sets cyc and cdk values as array
        #also the v values
        #and sets the fold points

        cycv = [self.startcyc]
        cdkv = [self.startcdk]
        vv = [ [self.startv[0], self.startv[1]] ]

        h=self.h
        totit = self.totit

        folds = []
        for i in range(totit):
            cyc0,cdk0,v0,v1 = self.predict(cycv[-1], cdkv[-1], vv[-1], h)
            cyc1,cdk1 = self.correct(cyc0,cdk0,v0,v1)
            cycv.append(cyc1)
            cdkv.append(cdk1)
            vv.append([v0,v1])

            #if the first component of v changes sign, we have a fold
            if vv[-1][0]*vv[-2][0] < 0:
                folds.append((cyc1,cdk1))

        self.cycv = np.array(cycv)
        self.cdkv = np.array(cdkv)
        self.vv = vv
        self.folds = folds

class SweepInputOneEnzyme(object):
    def __init__(self, **params):
        self.b = params['b']
        self.K = params['K']
        self.n = params['n']

        self.ap = params['ap']
        self.bp = params['bp']
        self.Kp = params['Kp']
        self.m = params['m']

        self.kXt = params['kXt'] #speed of Xt increase
        self.ka = params['ka'] #speed of a increase

        self.a0 = params['a0']
        self.Xt0 = params['Xt0']

    def a(self,t):
        return self.a0+self.ka*t
    def Xt(self, t):
        return self.Xt0+self.kXt*t

    def F(self,X,t):
        a = self.a(t)
        Xt = self.Xt(t)
        f = a + self.b*X**self.n/(self.K**self.n + X**self.n)
        g = self.ap + self.bp*self.Kp**self.m/(self.Kp**self.m + X**self.m)

        return f*(Xt-X) - g*X

    def solve(self,X0,T,dt):
        t=0
        X = X0
        Xv = []
        Xtv = []
        av = []
        tv = []
        while t<T:
            X = X + self.F(X,t)*dt
            Xv.append(X)
            tv.append(t)
            Xtv.append(self.Xt(t))
            av.append(self.a(t))

            t+=dt

        self.tv = np.array(tv)
        self.Xtv = np.array(Xtv)
        self.av = np.array(av)
        self.Xv = np.array(Xv)

    def getactivationtime(self,clo,chi):
        t1 = self.tv[np.nonzero(self.Xv>clo)[0][0]]
        t2 = self.tv[np.nonzero(self.Xv>chi)[0][0]]
        return t2-t1

class SweepCyclin(object):
    def __init__(self, **params):
        #parameters
        #first the degradation (APC/C) parameters
        self.ks = params['ks']
        self.adeg = params['adeg']
        self.bdeg = params['bdeg']
        self.ndeg = params['ndeg']
        self.Kdeg = params['Kdeg']

        #cdc25 loop
        self.acdc25 = params['acdc25']
        self.bcdc25 = params['bcdc25']
        self.ncdc25 = params['ncdc25']
        self.Kcdc25 = params['Kcdc25']

        #cdc25 is modified by cyclin activity
        self.cycfactor = params['cycfactor'] #this is a function

        #wee1 loop
        self.awee1 = params['awee1']
        self.bwee1 = params['bwee1']
        self.nwee1 = params['nwee1']
        self.Kwee1 = params['Kwee1']

    def cyclin(self,t):
        return self.ks*t
    def F(self,cdk,cyc):
        cf = self.cycfactor(cyc)
        cdc25 = cf*(self.acdc25 + self.bcdc25*cdk**self.ncdc25/(self.Kcdc25**self.ncdc25 + cdk**self.ncdc25))
        wee1 = self.awee1 + self.bwee1*self.Kwee1**self.nwee1/(self.Kwee1**self.nwee1 + cdk**self.nwee1)
        apc = self.adeg + self.bdeg*cdk**self.ndeg/(self.Kdeg**self.ndeg + cdk**self.ndeg)
        return self.ks + cdc25*(cyc-cdk) - wee1*cdk - apc*cdk

    def solve(self,cdk0,T,dt):
        t=0
        cdk=cdk0
        cdkv = []
        cycv = []
        tv = []
        while t<T:
            cdk = cdk + self.F(cdk,self.cyclin(t))*dt
            cdkv.append(cdk)
            tv.append(t)
            cycv.append(self.cyclin(t))
            t+=dt

        self.tv = np.array(tv)
        self.cycv = np.array(cycv)
        self.cdkv = np.array(cdkv)

    def getactivationtime(self,clo,chi):
        t1 = self.tv[np.nonzero(self.cdkv>clo)[0][0]]
        t2 = self.tv[np.nonzero(self.cdkv>chi)[0][0]]
        return t2-t1
