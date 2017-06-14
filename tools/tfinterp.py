#!/usr/bin/python -i
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import os
import math
import sys

def interp1d(xin,xp,yp):
    # this is a prototype of the tensorflow function
    x = np.clip(xin,xp.min(),xp.max())
    dx = xp[1]-xp[0]

    assert xp.shape==yp.shape,'xp and yp must be same size'
    assert len(xp.shape)==1,'only 1d interpolation'
    assert abs(np.diff(xp)/dx - 1.0).max() < 1e-6,'must be uniformly sampled'

    wt = np.maximum(0, 1-abs(np.reshape(x,x.shape+(1,)) - xp)/dx  )
    return (wt * yp).sum(axis=-1)


def interp1d_(xin_,xp,yp_):
    """
    Interpolate a uniformly sampled piecewise linear function. Mapping elements
    from xin_ to the result.  Input values will be clipped to range of xp.
        xin_ :  input tensor (real)
        xp : x grid (constant -- must be a 1d numpy array, uniformly spaced)
        yp_ : tensor of the result values at the gridpoints xp
    """
    import tensorflow as tf
    x_ = tf.clip_by_value(xin_,xp.min(),xp.max())
    dx = xp[1]-xp[0]
    assert len(xp.shape)==1,'only 1d interpolation'
    assert xp.shape[0]==int(yp_.get_shape()[0])
    assert abs(np.diff(xp)/dx - 1.0).max() < 1e-6,'must be uniformly sampled'

    newshape = [  ]
    x1_ = tf.expand_dims(x_,-1)
    dt = yp_.dtype
    wt_ = tf.maximum(tf.constant(0.,dtype=dt), 1-abs(x1_ - tf.constant(xp,dtype=dt))/dx  )
    y_ = tf.reduce_sum(wt_ * yp_,axis=-1)
    return y_


if __name__ == "__main__":
    # test
    xp = np.linspace(-1,1,5).astype(np.float32)
    yp = np.sin(xp*math.pi/2) 
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #SO LOUD!
    import tensorflow as tf

    x=np.random.uniform(-1.1,1.1,size=(3,4,5)).astype(np.float32)
    y = interp1d(x,xp,yp)

    if False:
        import matplotlib.pyplot as plt
        ix=x.ravel().argsort()
        plt.plot(x.ravel()[ix],y.ravel()[ix])

    y_ = interp1d_(tf.constant(x),xp,tf.constant(yp))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ycheck = sess.run(y_)
        assert np.allclose(y,ycheck)
    #execfile(os.environ['PYTHONSTARTUP'])
