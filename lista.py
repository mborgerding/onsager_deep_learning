#!/usr/bin/python -u
import tensorflow as tf
import numpy as np
import numpy.linalg as LA
import utils as ut
import math

st = ut.Setup()
st.flag('initTheta',0.1,'initial theta')
st.flag('untieS',False,'allow the S matrix to differ at each stage')
st.flag('setup',False,'initialize the last layer greedily')

##### create LISTA graph
# start with the input (to the LISTA inference algorithm)
y_ph = st.get_input_node()
print st

if st.setup:
    import lista_setup
    lista_setup.Setup(st)

Wd = st.noisy_Psi()

L = 1.1 * LA.norm(Wd,2)**2

We_ = ut.adjoint( Wd / L)
We = st.variable('We',We_)
We_ = st.loaded_state.get('We',We_)

def checkS(lS,dS):
    ls = lS.shape
    ds = dS.shape
    if ls== ds:
        return lS
    res = lS[:]
    if ls[0] < ds[0]:
        # expand with I-We_*Wd
        Snew = np.identity(st.n,dtype=Wd.dtype) - np.matmul(We_,Wd)
        ngrow = ds[0] - ls[0]
        tiledims = (ngrow,1,1)
        print 'appending S:=I-{We}*Wd  for new %d layers' % ngrow
        return np.append(lS, np.tile( Snew ,tiledims) ,0)
    else:
        return ut.broadcast_x_like_y(lS,dS)

thetaEps = 1e-3
def checkTheta(lt,dt):
    if len(lt) == len(dt):
        return lt
    elif len(lt)<len(dt):
        return np.float32( np.append(lt, np.tile( thetaEps ,  len(dt)-len(lt) ) ) )
    else:
        return lt[0:len(dt)]

S_ = np.identity(st.n,dtype=Wd.dtype) - 1.0/L* np.matmul(ut.adjoint(Wd),Wd)
if st.untieS is False:
    S = st.variable('S',S_)
    Svec = [S]*st.T
else:
    defaultS = np.zeros((st.T,st.n,st.n),dtype=np.float32)
    for i in range(st.T):
        defaultS[i,:,:] = S_
    Stensor = st.variable('S',defaultS)#,checkload=checkS)
    Svec = tf.split(0,st.T,Stensor)
    Svec = [tf.squeeze(S) for S in Svec]

theta = st.variable('theta',st.initTheta*np.float32( np.ones( st.T ) ))#,checkload=checkTheta)

#### build LISTA graph
B = tf.matmul( We , y_ph )
xhat = ut.softThreshold( B , theta[0])  # the zeroth layer
for t in range(1,st.T):  # layers 1 thru T-1
    xhat = ut.softThreshold( B + tf.matmul( Svec[t] ,xhat) , theta[t] )  # x(t) = h_theta( B + S*x(t-1) )

st.set_output_node(xhat)
print 'nmse_check : %.1fdB' % (10*math.log10(st.current_value(st.nmse_check)) )
st.run()

