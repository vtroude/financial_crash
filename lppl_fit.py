import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from numpy import cos, sin, log, exp, sqrt, pi, dot
from time import sleep
import itertools
from scipy.optimize import minimize
#from .lppl_garch import *

### UNITS ###

# dt in days
# ns in points
# t - vector of discrete time (0,1,2,...,N) 
# lnP - log price data 

### SET FIT PARAMETER BOUNDS ###
# bounds are set as global variables in the whole script
# thus, they do not need to be given as input to the functions

# constrained optimization tcb=(-90,365),mb=(0,1),wb=(1,50)
mb = (0.0,1.0)
wb = (1.,50.0) # should be changed to 2,15
tcb = (-90,365)

# unconstrained optimization
# mb = (-np.inf,np.inf)
# wb = (-np.inf,np.inf)
# # tcb = (-np.inf,np.inf)
# tcb = (0,np.inf)

# extended bounds 
# mb = (-1.0,2.0)
# wb = (4.,25.0)
# tcb = (-90,365)

### UNIMPORTANT ###

# rtcb = (0.9937,1.0063) # 0.1,10 growth per year
# rtcb = (0.1,10)
# Bhatb = (-5,5) # scaled B
# CBb = (-0.01,1.5) # oscillation to power law amplitude (0,1.5)
# C12b = (-1,1) # oscillation amplitude
# Db = (0.1,10) # damping

### STANDARD IN-SAMPLE LPPLS FITTING ###

### LPPL OLS FIT ###

def lppl_ols(t,lnP,tc,m,w,residuals=False):
	# given a triplet (tc,m,w), compute the OLS solution for the four linear parameters (A,B,C1,C2)
    # inputs must be 1d (squeezed) arrays
    
    # remove corrupt values 
    isfin = np.isfinite(lnP)
    t = t[isfin]
    lnP = lnP[isfin]

    # method 1: avoid computation of lppl(t) for t>=tc (undefined)
    # t = t[t<tc]
    # N = len(t) # sample size
    # lnP = lnP[:N]
    # tct = tc-t

    # method 2: use absolute value |tc-t| (standard method)
    tct = np.abs(tc-t)
    N = len(t)

    # compute features 
    arg = w * np.log(tct)
    g = np.power(tct,m)
    f1 = g * np.cos(arg)
    f2 = g * np.sin(arg)

    # remove irregular values
    inds = np.isfinite(arg) & np.isfinite(g) & np.isfinite(f1) & np.isfinite(f2)
    arg = arg[inds]
    g = g[inds]
    f1 = f1[inds]
    f2 = f2[inds]
    t = t[inds]
    lnP = lnP[inds]

    # assemble matrix (see paper filimonov & sornette, and others)
    MAT = np.array([[len(t), np.sum(g), np.sum(f1), np.sum(f2)],
                    [0, np.sum(g * g), np.sum(g * f1), np.sum(g * f2)],
                    [0, 0, np.sum(f1 * f1), np.sum(f1 * f2)],
                    [0, 0, 0, np.sum(f2 * f2)]])

    # matrix is symmetric
    MAT[1, 0] = MAT[0, 1]
    MAT[2, 0] = MAT[0, 2]
    MAT[3, 0] = MAT[0, 3]
    MAT[2, 1] = MAT[1, 2]
    MAT[3, 1] = MAT[1, 3]
    MAT[3, 2] = MAT[2, 3]

    # right hand side of matrix equation
    Y = np.array([np.sum(lnP), np.sum(g * lnP), np.sum(f1 * lnP), np.sum(f2 * lnP)])

    # solve ols by inversion 
    try:
        ABC = np.linalg.solve(MAT, Y)
    except np.linalg.LinAlgError:
        return [np.nan]*7,np.nan

    # linear parameter solution
    A,B,C1,C2 = ABC

    ### UNIMPORTANT ###

    # compute fit metaparameters
    # rtc = np.exp((A-lnP[-1])/(tc-t[-1])) # daily compound growth rate from [t2,tc] or [tc,t2] 
    # Bhat = B * (t[-1]-t[0]+1)
    # C = np.sqrt(C1**2+C2**2)
    # CB = np.abs(C/B)    
    # D = np.abs(m*B/(w*C))

    # reject solutions with linear parameters outside bounds
    # b1 = rtcb[0] <= rtc <= rtcb[1]
    # b2 = Bhatb[0] <= Bhat <= Bhatb[1]
    # b3 = CBb[0] <= CB <= CBb[1]
    # b4 = C12b[0] <= ABC[2] <= C12b[1]
    # b5 = C12b[0] <= ABC[3] <= C12b[1]
    # b6 = Db[0] <= D <= Db[1]
    # if not (b1 and b2 and b3 and b4 and b5 and b6):
    #     return [np.nan]*7,np.inf    

    # XXX additional OSC constraint? if CB>=0.05: OSC>=2.5 

    ###################

    # compute fit trajectory and objective function
    fit = A + B * g + C1 * f1 + C2 * f2
    resids = lnP - fit
    sse = np.sum(resids ** 2)/N*0.5 
    pars = ABC.tolist() + [tc,m,w]

    # AR(1) model component

    # correlation
    # rho = np.sum(resids[1:]*resids[:-1])/np.sum(resids[:-1]*resids[:-1])

    # uncorrelated residuals
    # ucresids = resids[1:] - rho * resids[:-1]

    # # GARCH(1,1) model component
    # _,optpars = fit_garch11_mle(ucresids,numpts=10,ncpus=6,gs=False)
    # a0,a1,a2 = optpars

    # # compute garch variance series 
    # sigmas = [a0]
    # for i in range(1,len(ucresids)):
    #     sigmas.append(a0 + a1*ucresids[i-1]**2 + a2*sigmas[-1])

    # # heteroskedastic, uncorrelated residuals
    # hucresids = ucresids/np.sqrt(sigmas)

    # ax=pd.Series(resids).plot();pd.Series(ucresids).plot(ax=ax)
    # pd.Series(sigmas).plot(ax=ax,secondary_y=True);plt.show()

    # compute new ssen 
    # ssen = np.var(ucresids)

    if residuals:
        return pars,resids

    return pars,sse

### LPPL WLS FIT ###

def lppl_wls(t,lnP,tc,m,w,weights):
    # XXX tc<t2 will lead to small tc, due to the different weighting

    # remove corrupt values 
    isfin = np.isfinite(lnP)
    t = t[isfin]
    lnP = lnP[isfin]
    weights = weights[isfin]

    # method 1: avoid computation of lppl(t) for t>=tc (undefined)
    # t = t[t<tc]
    # ns = len(t)
    # lnP = lnP[:ns]
    # if weights is not None: weights = weights[:ns]
    # tct = tc-t

    # method 2: use absolute value |tc-t|
    tct = np.abs(tc-t)

    # XXX normalize weights and set all to zero that are below 0.0001

    # compute only data points with non-zero weights
    select = weights!=0.
    weights = weights[select]
    t = t[select]
    lnP = lnP[select]
    ns = len(t)

    # compute features 
    arg = w * np.log(tct)
    f = np.power(tct,m)
    g = f * np.cos(arg)
    h = f * np.sin(arg)

    # remove irregular values
    inds = np.isfinite(arg) & np.isfinite(f) & np.isfinite(g) & np.isfinite(h)
    arg = arg[inds]
    f = f[inds]
    g = g[inds]
    h = h[inds]
    t = t[inds]
    lnP = lnP[inds]

    # assemble matrix
    MAT = np.array([[np.sum(weights), dot(weights,f), dot(weights,g), dot(weights,h)],
                    [0, dot(weights,f**2), np.sum(weights*f*g), np.sum(weights*f*h)],
                    [0, 0, dot(weights,g**2), np.sum(weights*g*h)],
                    [0, 0, 0, dot(weights,h**2)]])

    MAT[1, 0] = MAT[0, 1]
    MAT[2, 0] = MAT[0, 2]
    MAT[3, 0] = MAT[0, 3]
    MAT[2, 1] = MAT[1, 2]
    MAT[3, 1] = MAT[1, 3]
    MAT[3, 2] = MAT[2, 3]

    Y = np.array(
        [dot(weights,lnP), np.sum(weights*f*lnP), 
        np.sum(weights*g*lnP), np.sum(weights*h*lnP)]
        )

    # compute wls
    try:
        ABC = np.linalg.solve(MAT, Y)
    except np.linalg.LinAlgError:
        return [np.nan]*7,np.nan

    # compute metaparameters
    A,B,C1,C2 = ABC
    rtc = np.exp((A-lnP[-1])/(tc-t[-1])) # daily compound growth rate from [t2,tc] or [tc,t2] 
    Bhat = B * (t[-1]-t[0]+1)
    C = np.sqrt(C1**2+C2**2)
    CB = np.abs(C/B)    
    D = np.abs(m*B/(w*C))

    # reject solutions with linear parameters outside bounds
    b1 = rtcb[0] <= rtc <= rtcb[1]
    b2 = Bhatb[0] <= Bhat <= Bhatb[1]
    b3 = CBb[0] <= CB <= CBb[1]
    b4 = C12b[0] <= ABC[2] <= C12b[1]
    b5 = C12b[0] <= ABC[3] <= C12b[1]
    b6 = Db[0] <= D <= Db[1]
    if not (b1 and b2 and b3 and b4 and b5 and b6):
        return [np.nan]*7,np.inf    

    # compute fit and objective function
    fit = ABC[0] + ABC[1] * f + ABC[2] * g + ABC[3] * h
    ssen = 1./ns*np.sum(weights * (lnP - fit) ** 2)
    pars = ABC.tolist() + [tc,m,w]

    return pars,ssen

### GLS FIT ###
# NOT YET FINALIZED!!! # 
# returns weird results, check analytic equations and procedure

# def lppl_gls(t,lnP,tc,m,w,weights=None,plot=False):

#     # # initial OLS estimate
#     # pars,ssen = lppl_ols(t,lnP,tc,m,w)
#     # if np.isnan(pars).all():
#     #     return pars,ssen

#     # # compute OLS residuals
#     # A,B,C1,C2,tc,m,w = pars
#     # tct = np.abs(tc-t)
#     # lppl = A + B*(tct)**m + C1*(tct)**m*cos(w*log(tct)) + C2*(tct)**m*sin(w*log(tct))
#     # resids = lnP - lppl

#     if plot:
#         fig,ax = plt.subplots(1,1,figsize=(15,4))
#     #     pd.Series(resids).plot(ax=ax,c='k',label='OLS')

#     # # estimate pointwise variance of error terms and weights
#     # sigma2 = resids**2
#     # weights = 1./sigma2

#     # compute wls solution with inverse variance weights (heteroskedasticity)
#     pars,ssen = lppl_wls(t[1:],lnP[1:],tc,m,w,weights) # one point lost due to log-returns
#     if np.isnan(pars).all():
#         return pars,ssen

#     # compute WLS residuals
#     A,B,C1,C2,tc,m,w = pars
#     tct = np.abs(tc-t)
#     lppl = A + B*(tct)**m + C1*(tct)**m*cos(w*log(tct)) + C2*(tct)**m*sin(w*log(tct))
#     resids = lnP - lppl

#     if plot:
#         pd.Series(resids).plot(ax=ax,c='C0',label='WLS')

#     # estimate first-order autocorrelation coefficient
#     rho = np.corrcoef(np.array([resids[:-1], resids[1:]]))[0][1]

#     # transform data and perform FGLS regression

#     # remove corrupt values 
#     isfin = np.isfinite(lnP)
#     t = t[isfin]
#     lnP = lnP[isfin]

#     # use absolute value |tc-t|
#     tct = np.abs(tc-t)
#     N = len(t)

#     # compute features 
#     arg = w * np.log(tct)
#     g = np.power(tct,m)
#     f1 = g * np.cos(arg)
#     f2 = g * np.sin(arg)

#     # transform features
#     gt = g[1:] - rho * g[:-1]
#     f1t = f1[1:] - rho * f1[:-1]
#     f2t = f2[1:] - rho * f2[:-1]

#     # assemble matrix
#     MAT = np.array([[(1-rho)*len(t), (1-rho)*np.sum(gt), (1-rho)*np.sum(f1t), (1-rho)*np.sum(f2t)],
#                     [0, np.sum(gt * gt), np.sum(gt * f1t), np.sum(gt * f2t)],
#                     [0, 0, np.sum(f1t * f1t), np.sum(f1t * f2t)],
#                     [0, 0, 0, np.sum(f2t * f2t)]])

#     # symmetric matrix
#     MAT[1, 0] = MAT[0, 1]
#     MAT[2, 0] = MAT[0, 2]
#     MAT[3, 0] = MAT[0, 3]
#     MAT[2, 1] = MAT[1, 2]
#     MAT[3, 1] = MAT[1, 3]
#     MAT[3, 2] = MAT[2, 3]

#     # transform price
#     lnPt = lnP[1:] - rho * lnP[:-1]

#     # RHS of linear matrix equation
#     Y = np.array([np.sum(lnPt), np.sum(gt * lnPt), np.sum(f1t * lnPt), np.sum(f2t * lnPt)])

#     # solve equation
#     try:
#         ABC = np.linalg.solve(MAT, Y)
#     except np.linalg.LinAlgError:
#         return [np.nan]*7,np.nan

#     # compute metaparameters
#     A,B,C1,C2 = ABC
#     rtc = np.exp((A-lnP[-1])/(tc-t[-1])) # daily compound growth rate from [t2,tc] or [tc,t2] 
#     Bhat = B * (t[-1]-t[0]+1)
#     C = np.sqrt(C1**2+C2**2)
#     CB = np.abs(C/B)    
#     D = np.abs(m*B/(w*C))

#     # reject solutions with linear parameters outside bounds
#     b1 = rtcb[0] <= rtc <= rtcb[1]
#     b2 = Bhatb[0] <= Bhat <= Bhatb[1]
#     b3 = CBb[0] <= CB <= CBb[1]
#     b4 = C12b[0] <= ABC[2] <= C12b[1]
#     b5 = C12b[0] <= ABC[3] <= C12b[1]
#     b6 = Db[0] <= D <= Db[1]
#     if not (b1 and b2 and b3 and b4 and b5 and b6):
#         return [np.nan]*7,np.inf    

#     # XXX additional OSC constraint? if CB>=0.05: OSC>=2.5 

#     # compute fit trajectory and objective function
#     fit = A * (1-rho) + B * gt + C1 * f1t + C2 * f2t
#     resids = lnPt - fit
#     ssen = 1./N * np.sum(resids ** 2)
#     pars = ABC.tolist() + [tc,m,w]

#     # compute GLS residuals
#     A,B,C1,C2,tc,m,w = pars
#     tct = np.abs(tc-t)
#     lppl = A + B*(tct)**m + C1*(tct)**m*cos(w*log(tct)) + C2*(tct)**m*sin(w*log(tct))
#     resids = lnP - lppl

#     if plot:
#         pd.Series(resids).plot(ax=ax,c='C2',label='GLS')
#         plt.tight_layout()
#         ax.legend()
#         ax.set_title('Autocorrelation of OLS-Residuals: %0.2f'%rho)
#         plt.show()

#     return pars,ssen

### SUPPLEMENTARY FUNCTIONS FOR OLS AND WLS ###

def grid(t):
    t1,t2 = t[0],t[-1]
    dt = t2-t1
    tcu = t2+min(0.5*dt,tcb[1])
    tcr = np.linspace(0,tcu-1,5).tolist()
    # tcr = np.logspace(0,np.log10(365),5).round().astype(int).tolist()
    tcr = [t2 + i - 1 for i in tcr]
    mr = [0.2,0.4,0.6,0.8]
    wr = [5,10,15,20]
    gr = [list(x) for x in itertools.product(*[tcr,mr,wr])]
    return gr

# objective function (SSE(tc,m,w)) for OLS fitting
def cost_ssen(x,t,lnP):
    tc,m,w = x
    t1,t2 = t[0],t[-1]
    dt = t2-t1
    tcl = t2+max(-0.5*dt,tcb[0]) if np.isfinite(tcb[0]) else -np.inf
    tcu = t2+min(0.5*dt,tcb[1]) if np.isfinite(tcb[1]) else np.inf
    if not ((tcl<=tc<=tcu) and (mb[0]<=m<=mb[1]) and (wb[0]<=w<=wb[1])):
        # fits are allowed to touch the boundaries
        # these are later on filtered out
        return np.inf
    _,ssen = lppl_ols(t,lnP,tc,m,w)
    return ssen

def cost_wls(x,t,lnP,weights):
    tc,m,w = x
    t1,t2 = t[0],t[-1]
    dt = t2-t1
    tcl = t2+max(-0.5*dt,tcb[0]) if np.isfinite(tcb[0]) else -np.inf
    tcu = t2+min(0.5*dt,tcb[1]) if np.isfinite(tcb[1]) else np.inf
    if not ((tcl<=tc<=tcu) and (mb[0]<=m<=mb[1]) and (wb[0]<=w<=wb[1])):
        return np.inf
    _,ssen = lppl_wls(t,lnP,tc,m,w,weights)
    return ssen

def cost_gls(x,t,lnP,weights):
    tc,m,w = x
    t1,t2 = t[0],t[-1]
    dt = t2-t1
    tcl = t2+max(-0.5*dt,tcb[0]) if np.isfinite(tcb[0]) else -np.inf
    tcu = t2+min(0.5*dt,tcb[1]) if np.isfinite(tcb[1]) else np.inf
    if not ((tcl<=tc<=tcu) and (mb[0]<=m<=mb[1]) and (wb[0]<=w<=wb[1])):
        return np.inf
    _,ssen = lppl_gls(t,lnP,tc,m,w,weights)
    return ssen

# complete fit function: grid search + nonlinear minimizer + OLS / WLS / GLS
def lppl_fit_standard(t,lnP,weights=None,gls=False):
    # t[0] = t1 = 0

    # search grid over tc,m,w
    gr = grid(t)

    # select the cost function
    if gls: 
        cost = cost_gls
    elif weights is not None: 
        cost = cost_wls
    else:
        cost = cost_ssen

    # estimate weights from log-returns as centered moving variance (window = 11 points)
    if gls:
        lnr = np.diff(lnP,1)
        weights = np.array([1./np.var(lnr[max(0,i-5):min(len(lnr),i+5)]) for i in range(len(lnr))])

    # fit ols / wls
    args = (t,lnP) if weights is None else (t,lnP,weights)
    res = [minimize(cost,x,args=args,method='nelder-mead') for x in gr]

    # get (tc,m,w)-triplet minimizing average sse
    objs = [r.fun for r in res]
    mfun = min(objs)
    tc,m,w = res[objs.index(mfun)].x

    # XXX if tc*<t2, the data should be re-fit up to tc

    # compute optimal ols solution based on best (tc,m,w)
    if gls: 
        pars,sse = lppl_gls(t,lnP,tc,m,w,weights,plot=True)
    elif weights is not None:
        pars,sse = lppl_wls(t,lnP,tc,m,w,weights)
    else:
        pars,sse = lppl_ols(t,lnP,tc,m,w)

    return pars,sse

# convenience
def pars_to_df(pars,ssen):
    columns = ['A','B','C1','C2','tc','m','w','sse']
    return pd.DataFrame([pars+[ssen]],columns=columns)



























