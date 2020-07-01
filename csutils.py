#contains all kinds of functions to analyze a time series obtained by solving the changing switch system
import numpy as np

from scipy.signal import savgol_filter
from scipy.fftpack import fft,fftshift,ifft,ifftshift
from scipy.signal import sosfiltfilt, butter

def getperiodamplitude(tvals, xvals, threshold=1e-3):
    """gives the period and amplitude of the oscillations. Returns zero if no oscillations are seen.
       Oscillations with amplitude less than threshold are set to zero.
       We start from the last value. """

    def findnextmin(startindex):
        j=startindex
        while xvals[j] >= xvals[j+1]:
            if j+2>=len(xvals):
                #reach the end of the array
                return -1
            j +=1
        return j
    def findnextmax(startindex):
        i = startindex
        while xvals[i] <= xvals[i+1]:
            if i+2>=len(xvals):
                #reach the end of the array
                return -1
            i +=1
        return i
    xvals = xvals[-1::-1] #reverse
    if xvals[0] < xvals[1]: #start on an increasing piece
        nma = findnextmax(0)
        nmi = findnextmin(nma+1)
        nma2 = findnextmax(nmi+1)

        per = tvals[nma2] - tvals[nma]
        amp = xvals[nma] - xvals[nmi]
        if nma==-1 or nmi==-1 or nma2==-1:
            per, amp=0,0

        if abs(xvals[nma]-xvals[nma2])/amp>threshold:
            per,amp=0,0

    else:
        nmi = findnextmin(0)
        nma = findnextmax(nmi+1)
        nmi2 = findnextmin(nma+1)
        per = tvals[nmi2] - tvals[nmi]
        amp = xvals[nma] - xvals[nmi]
        if nma==-1 or nmi==-1 or nmi2==-1:
            per, amp=0,0
        if abs(xvals[nmi]-xvals[nmi2])/amp>threshold:
            per,amp=0,0

    per = abs(per)
    amp = abs(amp)
    return per,amp

def getarea(tv,xv,yv,per=2*np.pi):

    #returns the area of the curve in the x,y plane
    #note the input has to be periodic
    #first select one period (last one)?
    dt = tv[1]-tv[0]
    ind = int(per/dt)
    xc = xv[-ind:]
    yc = yv[-ind:]

    #derivative of y
    yder = (np.roll(yc,-1)-np.roll(yc,1))/2/dt

    return abs(dt*np.sum(xc*yder))


def getmeanmaxmin(tv,xv,per=2*np.pi):
    #returns the mean and amplitude
    #first select last period
    dt = tv[1]-tv[0]
    ind = int(per/dt)
    tc = tv[-ind:]
    xc = xv[-ind:]

    maxx = np.max(xc)
    minn = np.min(xc)
    mean = 1./per*np.sum(xc)*dt

    return mean,maxx,minn

def gettimeabove(tv,xv, thr, per=2*np.pi):
    #first select one period (last one)?
    dt = tv[1]-tv[0]
    ind = int(per/dt)
    xc = xv[-ind:]

    return np.sum(xc > thr)*dt

def gettimebelow(tv,xv,thr, per=2*np.pi):
    #first select one period (last one)?
    dt = tv[1]-tv[0]
    ind = int(per/dt)
    xc = xv[-ind:]

    return np.sum(xc > thr)*dt

def gettimebetween(tv, xv, thrlow, thrhigh, per):
    #purpose is to calculate the activation.transition rates

    if np.isclose(per, 0, atol=1e-2):
        return (0.,0.)

    dt = tv[1]-tv[0]

    #select one period, between the last two maxima
    ind = int(per/dt)

    Ma = np.argmax(xv[-ind:])

    xc = xv[Ma-2*ind:Ma-ind]
    tc = tv[Ma-2*ind:Ma-ind]

    try:
        #start down
        i=0
        while xc[i]>=thrhigh:
            i+=1
        mh=i
        while xc[i]>=thrlow:
            i+=1
        ml=i
        timedown = dt*(ml-mh)
        #next up:
        while xc[i] <= thrlow:
            i+=1
        ml=i
        while xc[i]<=thrhigh:
            i+=1
        mh=i
        timeup=dt*(mh-ml)
    except IndexError: #this means i went out of bounds, so take zero
        timeup=0
        timedown=0
    return (timeup, timedown)

def getdownupmaxspeed(tv,xv,per):
    #returns most negative and most positive value of derivative of xv
    dt =tv[1]-tv[0]

    tvs = tv[-int(per/dt):]
    xvs = xv[-int(per/dt):]

    dxs = (np.roll(xvs, -1) - np.roll(xvs,1))/ 2/dt
    return np.min(dxs), np.max(dxs)

def smooth_savgol(tv, xv, w=151, n=2):
    # use the Savitzky-Golay filter with window and degree given as arguments
    xs = savgol_filter(xv,w,n)
    ts = tv
    return ts, xs

def smooth_butter(tv, xv, n=2,co=1,fs=100):
    #butterworth filter
    sos = butter(n,co, output='sos',fs=fs)
    return tv, sosfiltfilt(sos, xv)

def getmaxima(tv,xv):
    #returns a list of all the local maxima
    mm = []
    i = 1
    while i < len(tv)-1:
        if xv[i] > xv[i-1] and xv[i] > xv[i+1]:
            mm.append((tv[i],xv[i]))
        i+=1
    return mm

def getminima(tv,xv):
    #returns a list of all the local minima
    mm = []
    i = 1
    while i < len(tv)-1:
        if xv[i] < xv[i-1] and xv[i] < xv[i+1]:
            mm.append((tv[i],xv[i]))
        i+=1
    return mm

def getinflectionpoint(xv,yv,startindex=1):
    #compute second derivative and check when it goes from pos to neg
    #note this assumes a sigmoid function
    ddy = (np.roll(yv,-1) - 2*yv+ np.roll(yv,1))/(xv[1]-xv[0])**2
    i=startindex
    while ddy[i] < 0 and i < len(ddy) :
        i+=1
    while i<len(ddy) and ddy[i]>0:
        i+=1
    if i >= len(ddy)-1:
        return (0,0)
    else:
        return xv[i],yv[i]

def getextrema(tv, xv):
    #returns maxima and minima as one array
    mm = []
    i = 1
    while i < len(tv)-1:
        if xv[i] < xv[i-1] and xv[i] < xv[i+1]:
            mm.append((tv[i],xv[i], 'min'))
        elif xv[i] > xv[i-1] and xv[i] > xv[i+1]:
            mm.append((tv[i],xv[i], 'max'))
        i+=1
    return mm


def getcrossingtimes(tv, xv, threshold):
    # return times point at which x crosses threshold from below
    i = 0
    ct = []
    while i < len(tv)-1:
        if xv[i] < threshold and xv[i+1] > threshold:
            ct.append(0.5*(tv[i] + tv[i+1]))
        i+=1
    return ct

def getcrossingtimes_twothresholds(tv,xv,thrup,thrdown):
    # detects the time points where the time series crosses thrup from low to high
    # and thrdown from high to low
    i = 0
    ct = []
    # note: the elements of the list are tuples (t, dir) where dir is a character u or d
    while i < len(tv)-1:
        if xv[i] < thrup and xv[i+1] > thrup:
            ct.append((0.5*(tv[i] + tv[i+1]), 'u'))
        if xv[i] > thrdown and xv[i+1] < thrdown:
            ct.append((0.5*(tv[i] + tv[i+1]), 'd'))
        i+=1
    return ct
