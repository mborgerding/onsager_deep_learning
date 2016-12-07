#!/usr/bin/python -u
import numpy as np
import math
import sys

__doc__="""
    estimate X from Y=A*X+W,
    where
        X is a sparse N-length vector
        A is a known MxN matrix
        W is Gaussian noise vector length M
        Typically M << N

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
st.flag('shrink','bg','threshold type. One of {"soft","bg","pwlin","expo","spline"} ')
st.flag('errTheta',0.0,'perturb the parameters from the loaded solution by adding the given value(s)')
st.flag('errV',0.0,'perturb the parameters from the loaded solution by adding the given value(s)')
st.flag('errISU',0.0,'perturb the parameters from the loaded solution by adding the given value(s)')
st.flag('errRS2',0.0,'perturb the parameters from the loaded solution by adding the given value(s)')
st.flag('errYvar',0.0,'perturb the parameters from the loaded solution by adding the given value(s)')

# start with the input to the inference algorithm
y_ph = st.get_input_node()
print st
M,N,L = (st.m,st.n,st.mbs)

# choose the appropriate shrinkage function and some ballpark initialization parameters
shrink_func,theta_def = shr.get_shrinkage_function(st.shrink)
print 'shrink_func=%s theta_def=%s' % ( repr(shrink_func) , repr(theta_def))

I_m = np.float32( np.identity(M) )

ls = st.loaded_state
st.loaded_state = {}

# perturb  a numpy array and return a tf variable for it
def perturbed_variable(name,x,mag,dt=np.float32):
    y = np.array( x + np.sign(np.random.uniform(-1,1,size=np.shape(x)) ) * mag).astype(dt)
    if mag:
        print 'perturbing %s with a Rademacher signal of value +-%g'% (name,mag)
        #print '%s from %s to %s' % (name,repr(x),repr(y))
    return st.variable(name,y)

# The number of matrices learned depends on whether we are doing VAMP(VAMP) or LAMP, and whether LAMP has untied matrices
n_matrices=0 # S form is prescribed by VAMP
need_parms = ['theta_%d'%t for t in range(st.T) ]

lsk = ls.keys()
if all([k in lsk for k in need_parms]):
    print 'refining learned parameters from ' + st.load
    theta = [ perturbed_variable('theta_%d'%t,ls['theta_%d'%t],st.errTheta) for t in range(st.T) ]
else:
    print 'starting from default parameters'
    theta = [ perturbed_variable('theta_%d'%t, theta_def,st.errTheta) for t in range(st.T) ]

# do an SVD in numpy
A = st.Psi
AA = np.matmul(A,A.T)
s2,U = la.eigh(AA)  # this might be bad if kappa is extremely high
s = np.sqrt(s2)
V = ut.diagMult( 1/s, np.matmul( U.T,A)).T

# precompute some tensorflow constants
rS2 = perturbed_variable('rS2', np.reshape( 1/(s*s),(-1,1) ) , st.errRS2)  # reshape to (M,1) to allow broadcasting
rj = tf.zeros( (N,L) ,dtype=y_ph.dtype)
taurj =  tf.reduce_sum(y_ph*y_ph,0)/(N)
A = st.Psitf
yvar = perturbed_variable( 'yvar',st.noise_var,st.errYvar )

iSU = ut.diagMult(1/s,U.T).astype(np.float32)
iSU_ = perturbed_variable('iSU',iSU,st.errISU)
ytilde = tf.matmul( iSU_,y_ph)  # inv(S)*U*y

V_ = perturbed_variable('V', V.astype(np.float32),st.errV )
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

st.set_output_node(xhat)

print 'nmse_check : %.1fdB' % (10*math.log10(st.current_value(st.nmse_check)) )
st.run()
# vim: set ts=4 et sw=4 ft=python ai:
