#!/usr/bin/python
import tensorflow as tf
import numpy as np
import math
import utils as ut
la = ut.la

def eta_opt_lambda(rhat,x,**kwargs):
    'find the MSE-optimal lambda'

    minlam = 0
    maxlam = np.abs(rhat).max()

    for k in range(5):
        lamvec = np.linspace(minlam,maxlam,11)
        dlam = lamvec[1] - lamvec[0]
        err = [ la.norm( x - ut.eta(rhat,lam)) for lam in lamvec]
        minidx = np.argmin(err)
        bestlam = lamvec[minidx]
        minlam = max(0,bestlam - dlam)
        maxlam = bestlam + dlam
    xhat = ut.eta(rhat,bestlam)
    #print 'lambda = %.3f nmse=%.3fdB' % (bestlam,20*math.log10( la.norm(err[minidx])/la.norm(x) ) )
    return (xhat,bestlam)


def bg_gen(A,L=1000,pnz=.1,snr=40):
    M,N = A.shape
    X = (np.random.uniform(size=(N,L)) < pnz) * np.random.normal(size=(N,L))
    AX = np.matmul(A,X)
    Y = AX + np.random.normal( size=(M,L), scale=la.norm(AX)/math.sqrt(AX.size) * 10**(snr/-20) )
    return X,Y

def lista_run(y,We,S,theta,T,**kwargs):
    B = np.matmul(We,y)
    xhat = ut.eta(B,theta[0])
    for t in range(1,T):
        xhat = ut.eta(B + np.matmul( S[t], xhat ) ,theta[t])
    return xhat

def greedyUntied(st):
    A = st.Psi
    M,N = A.shape
    mul = np.matmul
    ls = st.loaded_state
    We = ls['We']
    S = ls['S']
    theta=ls['theta']
    L = 5000

    while len(S) < st.T:
        Tm1 = len(S)
        T = Tm1 + 1

        x,y = bg_gen(A,L)
        B = mul(We,y)
        xhat = lista_run(y,We,S,theta,Tm1)

        S_ = np.identity(N) - mul(We,A)
        theta_ = theta[Tm1-1]

        stepsize = 1.0
        nmsePrev = 999
        for steps in range(10):
            rhat_ = B + mul(S_,xhat)
            xhat_ = ut.eta(rhat_ ,theta_)
            nmse = 20*math.log10(la.norm(x-xhat_,'fro')/la.norm(x,'fro'))
            print ' %d nmse=%.4fdB ' % ( steps, nmse)
            if nmse > nmsePrev:
                stepsize = stepsize * .5
            else:
                stepsize = stepsize * 1.1
            nmsePrev = nmse
            S_ = S_ - stepsize * mul( ((xhat_-x)*abs(np.sign(xhat_))), rhat_.T) /L
            theta_ = theta_ - stepsize * np.mean( (xhat_-x)*(-np.sign(xhat_)) )
        xhat = ut.eta(B + mul(S_,xhat) ,theta_)
        S = np.concatenate( (S,np.reshape(S_,(1,N,N)) ) )
        theta = np.concatenate( (theta,np.reshape(theta_,(1,)) ))
        print 't=%d nmse=%.2fdB ' % (Tm1, 20*math.log10(la.norm(xhat-x)/la.norm(x) ) )
    x,y = bg_gen(A,L)
    xhat = lista_run(y,We,S,theta,st.T)
    print 'fresh nmse=%.2fdB' % ( 20*math.log10(la.norm(xhat-x)/la.norm(x) ) )

    return S,theta


def Setup(st,**kwargs):
    A = st.Psi
    M,N = A.shape
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer())
        (y, x) = sess.run(st.generators)

    mul = np.matmul

    xhat = np.zeros_like(x)

    ls = st.loaded_state
 
    for key in ('We','S','theta'):
        print '%s.shape = %s' %(key,repr(ls[key].shape))

    if st.untieS:
        S,theta = greedyUntied(st)
    else:
        We = ls['We']
        S = ls['S']
        theta=ls['theta']
        Tprev = len(theta)
        theta = np.concatenate( (theta,np.zeros(st.T-Tprev)))
        bt=0
        v=0

        B = mul( We , y )

        for t in range(st.T):
            # basic recursion:
            # xhat = eta(We*y + S*xhat;theta_t)
            rhat = B + mul( S ,xhat)
            if t < Tprev:
                xhat = ut.eta(rhat,theta[t])
            else:
                (xhat,theta[t]) = eta_opt_lambda( rhat ,x)
            print 't=%d lambda=%.3f nmse=%.3fdB' % ( t,theta[t],20*math.log10( la.norm(xhat-x)/la.norm(x) ) )

    ls['theta'] = np.float32(theta)
    ls['S'] = np.float32( S)
    return theta,S

if __name__=="__main__":
    st = ut.Setup()  # the common structure of all the ISTA,LISTA,AMP experiments
    st.flag('untieS',False,'allow each layer to have its own dense matrix')
    y_ph = st.get_input_node()
    Setup(st)

