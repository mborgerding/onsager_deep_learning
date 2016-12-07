#!/usr/bin/python -u
import numpy as np
import math
import sys

__doc__="""
This script implements the LAMP[1] or LVAMP[2 + learning] algorithm with choice of shrinkage function.

Both algorithms aims at solving the Sparse Linear Inverse Problem

    estimate X from Y=A*X+W,
    where
        X is a sparse N-length vector
        A is a known MxN matrix
        W is Gaussian noise vector length M
        Typically M << N

LAMP does so by optimizing _theta_,_S_ within the recursive ( or layerwise ) algorithm

for t in 0..T-1
    b[t] = (N/M) * dxdr[t-1]
    v[t] = y - A * xhat[t-1]  +  b[t]*v[t-1]
    rvar[t] = norm(v[t])**2 / M
    (xhat[t],dxdr[t]) = shrink_func( xhat[t-1] + A^T * _S_[t]*v[t], rvar[t] , _theta_[t] )

The learned variables are
    _theta_ in Real^(TxK) ( K depends on shrinkage function )
    _S_ in  { Real^(MxM) if --tieS=True }
         or { Real^(TxMxM) if --tieS=False }

LVAMP does so by optimizing  _theta_ within the recursive ( or layerwise ) algorithm

for t in 0..T-1
    forall n<N,k<L
        scale[n,k] =  1/( 1 + yvar/taurj[k]/(s[n]^2) )
    for k<L
        zeta[k] = (N/M)/mean(scale[:,k])
    ri = ( V * (scale .* (U'*y - V'*rj) ) * diag(zeta))
    for k<L
        tauri[k] = taurj[k]*(zeta[k]-1)
    xhat,dxdr = shrink_func( ri, tauri, _theta_[t] )
    rj = (xhat - dxdr*ri)/(1-dxdr)
    taurj = tauri * dxdr/(1-dxdr)

    Where A = U*diag(s)*V' by SVD

For more info, see:
[1] Mark Borgerding, Philip Schniter. "Onsager-corrected deep learning for sparse linear inverse problems." arXiv:1607.05966
[2] Alyson Fletcher, Mojtaba Sahraee-Ardakan, Sundeep Rangan, Philip Schniter. "Expectation Consistent Approximate Inference: Generalizations and Convergence" arXiv:1602.07795

Run this script with --help to see all arguments.
"""
if len(sys.argv)==1:
    sys.stderr.write(__doc__)
    sys.exit(1)

import tensorflow as tf
import utils as ut
import shrinkage as shr
la = ut.la
# set up the inverse problem basic structure
# y = A*x + w
st = ut.Setup()  # the common structure of all the ISTA,LISTA,AMP experiments
st.flag('tieS',True,"tie all layers' S matrices to be identical")
st.flag('shrink','soft','threshold type. One of {"soft","bg","pwlin","expo","spline"} ')
st.flag('vamp',False,'Use the VAMP algorithm as a foundation rather than AMP')
st.flag('perturb','','perturb the parameters from the loaded solution by adding the given value(s)')
st.flag('T1weight',0,'In the case of T=1, the shrinkage function is overparameterized.  Add a small penalty on norm(v_1) in order to mitigate this. ')
st.flag('matched',False,'use the nominal values prescribed by VAMP')

# start with the input to the inference algorithm
y_ph = st.get_input_node()

if st.matched:
    st.cfg.shrink='bg'
    st.cfg.vamp=True
    st.cfg.reportEvery=2
    st.cfg.refinements=0
    st.cfg.trainRate=0
    st.cfg.stopAfter=2

print st
M,N,L = (st.m,st.n,st.mbs)

# choose the appropriate shrinkage function and some ballpark initialization parameters
shrink_func,theta_def = shr.get_shrinkage_function(st.shrink)
print 'shrink_func=%s theta_def=%s' % ( repr(shrink_func) , repr(theta_def))

I_m = np.float32( np.identity(M) )
ls = st.loaded_state

# The number of matrices learned depends on whether we are doing VAMP(VAMP) or LAMP, and whether LAMP has untied matrices
if st.vamp:
    n_matrices=0 # S form is prescribed by VAMP
    need_parms = ['yvar']
elif st.tieS:
    n_matrices=1
    need_parms = ['S_0']
else:
    n_matrices=st.T
need_parms = ['S_%d' %t for t in range(n_matrices)] + ['theta_%d'%t for t in range(st.T) ]

lsk = ls.keys()
if all([k in lsk for k in need_parms]):
    print 'have all %d layers needed -- performing whole-system fine-tuning' % (st.T)

    # perturb the loaded parameters
    if len(st.perturb):
        ntheta = len(ls['theta_0'])
        #for t in range(st.T):
        t = 0
        ls['theta_%d'%t] = ls['theta_%d'%t] + np.array(eval(st.perturb)).astype(np.float32)

    if not st.vamp:
        if st.tieS:
            S = [ st.variable('S_0',ls['S_0']) ] # only one S entry needed
        else:
            S = [ st.variable('S_%d'%t,ls['S_%d'%t]) for t in range(st.T) ]
    theta = [ st.variable('theta_%d'%t,ls['theta_%d'%t]) for t in range(st.T) ]
else:
    # learning *some* of the parameters
    print 'Learning some of the parameters,specifically the %s ' % repr(set(need_parms)-set(lsk))
    assert(st.perturb == '')

    S = [None]*n_matrices
    S_def = I_m
    for t in range(n_matrices):
        k = 'S_%d' % t
        if ls.has_key(k):
            S[t] = tf.constant(ls[k] ,dtype=tf.float32,name=k)
            S_def = ls[k]
        else:
            S[t] = st.variable(k,S_def)

    theta = [None]*st.T
    for t in range(st.T):
        k = 'theta_%d'%t
        if ls.has_key(k):
            theta[t] = tf.constant(ls[k] ,dtype=tf.float32,name=k)
            if not (t==0 and st.shrink=='soft'):
                theta_def = ls[k]
        else:
            # The scaled soft-threshold is over-parameterized for T=1 LAMP
            # Use a unity gain for beta_0 (aka theta[1] )
            if t==0 and st.shrink=='soft':
                theta[t] = st.variable(k,theta_def[0] )
            else:
                theta[t] = st.variable(k,theta_def )

print 'learning variables:%s' % repr(st.variables.keys())

if st.vamp:
    # do an SVD in numpy
    A = st.Psi
    AA = np.matmul(A,A.T)
    s2,U = la.eigh(AA)  # this might be bad if kappa is extremely high
    s = np.sqrt(s2)
    V = ut.diagMult( 1/s, np.matmul( U.T,A)).T

    # precompute some tensorflow constants
    rS2 = tf.constant( np.reshape( 1/(s*s),(-1,1) ).astype(np.float32) )  # reshape to (M,1) to allow broadcasting
    rj = tf.zeros( (N,L) ,dtype=y_ph.dtype)
    taurj =  tf.reduce_sum(y_ph*y_ph,0)/(N)
    A = st.Psitf
    if st.matched:
        yvar = tf.constant(st.noise_var,dtype=np.float32)
    else:
        yvar = tf.exp( st.variable( 'logyvar', 0.0) ) # use a log domain variable for tracking yvar to prevent non-positive variances
    ytilde = tf.matmul( tf.constant( ut.diagMult(1/s,U.T).astype(np.float32) ) ,y_ph)  # inv(S)*U*y
    V_ = tf.constant(V.astype(np.float32))
    V_T = tf.transpose(V_)

    xhat = tf.constant(0,dtype=tf.float32)
    for t in range(st.T):  # layers 0 thru T-1
        # linear step (LMMSE estimation and Onsager correction)
        varRat = tf.reshape(yvar/taurj,(1,-1) ) # one per column
        scaleEach = 1/( 1 + rS2*varRat ) # LMMSE scaling individualized per element {singular dimension,column}
        zetai = st.n/tf.reduce_sum(scaleEach,0) # one per column  (zetai is 1/(1-alphai) from Phil's derivation )
        adjust = ut.diagMult( scaleEach*(ytilde - tf.matmul(V_T,rj)) , zetai) #  adjustment in the s space
        ri = rj + tf.matmul(V_, adjust )  # bring the adjustment back into the x space and apply it
        tauri = taurj*(zetai-1) # adjust the variance

        # non-linear step
        xhat,dxdr = shrink_func(ri,tauri,theta[t])
        zetaj = 1/(1-dxdr)
        rj = (xhat - dxdr*ri)*zetaj # apply Onsager correction
        taurj = tauri*(zetaj-1) # adjust the variance
else:
    A = st.Psitf
    xhat = tf.constant(0,dtype=tf.float32)
    MUL = tf.matmul
    NoverM = tf.constant(float(N)/M,dtype=tf.float32)
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)

    for t in range(st.T):  # layers 0 thru T-1
        if t==0:
            # xhat is zero, vt is zero
            vt = y_ph
        else:
            bt = dxdr * NoverM
            vt = y_ph - MUL( A , xhat ) + bt * vt
        rvar = tf.reduce_sum(tf.square(vt),0) * OneOverM
        if t==0 or not st.tieS:
            Bt = MUL(ut.adjoint(A) ,S[t] )
        (xhat,dxdr) = shrink_func( xhat + MUL(Bt,vt),rvar , theta[t] )

    # in the case of T==1, at least some of the shrinkage functions  (e.g. "soft")
    # are overparameterized for only a loss on xhat.
    # To mitigate this, we add an extremely weak penalty on the norm of the next v
    if st.T == 1 and st.T1weight > 0:
        bt = dxdr * NoverM
        vt = y_ph - MUL( A , xhat ) + bt * vt
        rvar = tf.reduce_sum(tf.square(vt),0) * OneOverM
        st.add_penalty(st.T1weight * rvar )

st.set_output_node(xhat)

print 'nmse_check : %.1fdB' % (10*math.log10(st.current_value(st.nmse_check)) )
st.run()
# vim: set ts=4 et sw=4 ft=python ai:
