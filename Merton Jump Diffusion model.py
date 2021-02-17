
#Merton Jump Diffusion- Analytical and Simulation vs Black Scholes

import math as math
import numpy as np

def sn_cdf(x):
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)/math.sqrt(2.0)

    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)

    return 0.5*(1.0 + sign*y)


def BS_call(S0,sig, tau,r, K):
    d1=math.log(S0/K)+(r+0.5*sig*sig)*tau
    d1/=(sig*math.sqrt(tau))
    d2=d1-sig*math.sqrt(tau)

    price=S0*sn_cdf(d1)-K*math.exp(-r*tau)*sn_cdf(d2)
    return price

def merton_call(S0,sig, tau,r, K,lam,mu_y,sig_y, N):
    price=0.0
    k=math.exp(mu_y+0.5*sig_y*sig_y)-1
    for n in range(N):
        sig_n=math.sqrt( sig*sig+n*sig_y*sig_y/tau)
        S0_n=S0*math.exp(-lam*k*tau+n*mu_y+0.5*n*sig_y*sig_y)
        prob_n=(lam*tau)**n/math.factorial(n)*math.exp(-lam*tau)
        price+=BS_call(S0_n,sig_n, tau,r, K)*prob_n
    return price
        
def merton_call_alt(S0,sig, tau,r, K,lam,mu_y,sig_y, N):
    price=0.0
    k=math.exp(mu_y+0.5*sig_y*sig_y)-1
    lam_b=lam*(1.0+k)
    for n in range(N):
        sig_n=math.sqrt( sig*sig+n*sig_y*sig_y/tau)
        r_n=r-lam*k+n*(mu_y+0.5*sig_y*sig_y)/tau
        prob_n=(lam_b*tau)**n/math.factorial(n)*math.exp(-lam_b*tau)
        price+=BS_call(S0,sig_n, tau,r_n, K)*prob_n
    return price

def merton_call_simulation(S0,sig, tau,r, K,lam,mu_y,sig_y):
    nsim=100000
    k=math.exp(mu_y+0.5*sig_y*sig_y)-1
    price=0.0
    for i in range(nsim):
        jump_N=np.random.poisson(lam*tau)
        jump_normal=np.random.normal(mu_y,sig_y,jump_N)
        jump_sum=np.sum(jump_normal)
        S_tau=S0*math.exp((r-lam*k-0.5*sig*sig)*tau+sig*np.random.normal(0,math.sqrt(tau))+jump_sum)
        price+=max(S_tau-K,0)
    price*=math.exp(-r*tau)
    price/=nsim
    return price


print(BS_call(100,0.4, 1,0.05, 100))
print(merton_call(100,0.4, 1,0.05, 100,0.5, 0.1,0.1, 20))
print(merton_call_alt(100,0.4, 1,0.05, 100,0.5, 0.1,0.1, 20))
print(merton_call_simulation(100,0.4, 1,0.05, 100,0.5, 0.1,0.1)) 