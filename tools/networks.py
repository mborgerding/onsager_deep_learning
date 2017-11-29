#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la

import tensorflow as tf
import tools.shrinkage as shrinkage

def build_LISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.simple_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_)
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ )
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) )
    return layers


def build_LAMP(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LVAMP(prob,T,shrink):
    """
    Build the LVMAP network with an SVD parameterization.
    Learns the measurement noise variance and nonlinearity parameters
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    AA = np.matmul(A,A.T)
    s2,U = la.eigh(AA)  # this is faster than svd, but less precise if ill-conditioned
    s = np.sqrt(s2)
    V = np.matmul( A.T,U) / s
    print('svd reconstruction error={nmse:.3f}dB'.format(nmse=20*np.log10(la.norm(A-np.matmul(U*s,V.T))/la.norm(A) ) ) )
    assert np.allclose( A, np.matmul(U*s,V.T),rtol=1e-4,atol=1e-4)
    V_ = tf.constant(V,dtype=tf.float32,name='V')

    # precompute some tensorflow constants
    rS2_ = tf.constant( np.reshape( 1/(s*s),(-1,1) ).astype(np.float32) )  # reshape to (M,1) to allow broadcasting
    #rj_ = tf.zeros( (N,L) ,dtype=tf.float32)
    rj_ = tf.zeros_like( prob.x_)
    taurj_ =  tf.reduce_sum(prob.y_*prob.y_,0)/(N)
    logyvar_ = tf.Variable( 0.0,name='logyvar',dtype=tf.float32)
    yvar_ = tf.exp( logyvar_)
    ytilde_ = tf.matmul( tf.constant( ((U/s).T).astype(np.float32) ) ,prob.y_)  # inv(S)*U*y
    Vt_ = tf.transpose(V_)

    xhat_ = tf.constant(0,dtype=tf.float32)
    for t in range(T):  # layers 0 thru T-1
        # linear step (LMMSE estimation and Onsager correction)
        varRat_ = tf.reshape(yvar_/taurj_,(1,-1) ) # one per column
        scale_each_ = 1/( 1 + rS2_*varRat_ ) # LMMSE scaling individualized per element {singular dimension,column}
        zetai_ = N/tf.reduce_sum(scale_each_,0) # one per column  (zetai_ is 1/(1-alphai) from Phil's derivation )
        adjust_ = ( scale_each_*(ytilde_ - tf.matmul(Vt_,rj_))) * zetai_ #  adjustment in the s space
        ri_ = rj_ + tf.matmul(V_, adjust_ )  # bring the adjustment back into the x space and apply it
        tauri_ = taurj_*(zetai_-1) # adjust the variance

        # non-linear step
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        xhat_,dxdr_ = eta(ri_,tauri_,theta_)
        if t==0:
            learnvars = None # really means "all"
        else:
            learnvars=(theta_,)
        layers.append( ('LVAMP-{0} T={1}'.format(shrink,t+1),xhat_, learnvars ) )

        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        zetaj_ = 1/(1-dxdr_)
        rj_ = (xhat_ - dxdr_*ri_)*zetaj_ # apply Onsager correction
        taurj_ = tauri_*(zetaj_-1) # adjust the variance

    return layers

def build_LVAMP_dense(prob,T,shrink,iid=False):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    layers=[]
    A = prob.A
    M,N = A.shape

    Hinit = np.matmul(prob.xinit,la.pinv(prob.yinit) )
    H_ = tf.Variable(Hinit,dtype=tf.float32,name='H0')
    xhat_lin_ = tf.matmul(H_,prob.y_)
    layers.append( ('Linear',xhat_lin_,None) )

    if shrink=='pwgrid':
        theta_init = np.linspace(.01,.99,15).astype(np.float32)
    vs_def = np.array(1,dtype=np.float32)
    if not iid:
        theta_init = np.tile( theta_init ,(N,1,1))
        vs_def = np.tile( vs_def ,(N,1))

    theta_ = tf.Variable(theta_init,name='theta0',dtype=tf.float32)
    vs_ = tf.Variable(vs_def,name='vs0',dtype=tf.float32)
    rhat_nl_ = xhat_lin_
    rvar_nl_ = vs_ * tf.reduce_sum(prob.y_*prob.y_,0)/N

    xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
    layers.append( ('LVAMP-{0} T={1}'.format(shrink,1),xhat_nl_, None ) )
    for t in range(1,T):
        alpha_nl_ = tf.reduce_mean( alpha_nl_,axis=0) # each col average dxdr

        gain_nl_ = 1.0 /(1.0 - alpha_nl_)
        rhat_lin_ = gain_nl_ * (xhat_nl_ - alpha_nl_ * rhat_nl_)
        rvar_lin_ = rvar_nl_ * alpha_nl_ * gain_nl_

        H_ = tf.Variable(Hinit,dtype=tf.float32,name='H'+str(t))
        G_ = tf.Variable(.9*np.identity(N),dtype=tf.float32,name='G'+str(t))
        xhat_lin_ = tf.matmul(H_,prob.y_) + tf.matmul(G_,rhat_lin_)

        layers.append( ('LVAMP-{0} lin T={1}'.format(shrink,1+t),xhat_lin_, (H_,G_) ) )

        alpha_lin_ = tf.expand_dims(tf.diag_part(G_),1)

        eps = .5/N
        alpha_lin_ = tf.maximum(eps,tf.minimum(1-eps, alpha_lin_ ) )

        vs_ = tf.Variable(vs_def,name='vs'+str(t),dtype=tf.float32)

        gain_lin_ = vs_ * 1.0/(1.0 - alpha_lin_)
        rhat_nl_ = gain_lin_ * (xhat_lin_ - alpha_lin_ * rhat_lin_)
        rvar_nl_ = rvar_lin_ * alpha_lin_ * gain_lin_

        theta_ = tf.Variable(theta_init,name='theta'+str(t),dtype=tf.float32)

        xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
        alpha_nl_ = tf.maximum(eps,tf.minimum(1-eps, alpha_nl_ ) )
        layers.append( ('LVAMP-{0}  nl T={1}'.format(shrink,1+t),xhat_nl_, (vs_,theta_,) ) )

    return layers
