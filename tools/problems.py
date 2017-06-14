#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf

class Generator(object):
    def __init__(self,A,**kwargs):
        self.A = A
        M,N = A.shape
        vars(self).update(kwargs)
        self.x_ = tf.placeholder( tf.float32,(N,None),name='x' )
        self.y_ = tf.placeholder( tf.float32,(M,None),name='y' )

class TFGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)
    def __call__(self,sess):
        'generates y,x pair for training'
        return sess.run( ( self.ygen_,self.xgen_ ) )

class NumpyGenerator(Generator):
    def __init__(self,**kwargs):
        Generator.__init__(self,**kwargs)

    def __call__(self,sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches,self.nsubprocs)


def bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=None,SNR=40):

    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)
    if kappa >= 1:
        # create a random operator with a specific condition number
        U,_,V = la.svd(A,full_matrices=False)
        s = np.logspace( 0, np.log10( 1/kappa),M)
        A = np.dot( U*(s*np.sqrt(N)/la.norm(s)),V).astype(np.float32)
    A_ = tf.constant(A,name='A')
    prob = TFGenerator(A=A,A_=A_,pnz=pnz,kappa=kappa,SNR=SNR)
    prob.name = 'Bernoulli-Gaussian, random A'

    bernoulli_ = tf.to_float( tf.random_uniform( (N,L) ) < pnz)
    xgen_ = bernoulli_ * tf.random_normal( (N,L) )
    noise_var = pnz*N/M * math.pow(10., -SNR / 10.)
    ygen_ = tf.matmul( A_,xgen_) + tf.random_normal( (M,L),stddev=math.sqrt( noise_var ) )

    prob.xval = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yval = np.matmul(A,prob.xval) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    prob.xinit = ((np.random.uniform( 0,1,(N,L))<pnz) * np.random.normal(0,1,(N,L))).astype(np.float32)
    prob.yinit = np.matmul(A,prob.xinit) + np.random.normal(0,math.sqrt( noise_var ),(M,L))
    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.noise_var = noise_var

    return prob

def random_access_problem(which=1):
    import raputil as ru
    if which == 1:
        opts = ru.Problem.scenario1()
    else:
        opts = ru.Problem.scenario2()

    p = ru.Problem(**opts)
    x1 = p.genX(1)
    y1 = p.fwd(x1)
    A = p.S
    M,N = A.shape
    nbatches = int(math.ceil(1000 /x1.shape[1]))
    prob = NumpyGenerator(p=p,nbatches=nbatches,A=A,opts=opts,iid=(which==1))
    if which==2:
        prob.maskX_ = tf.expand_dims( tf.constant( (np.arange(N) % (N//2) < opts['Nu']).astype(np.float32) ) , 1)

    _,prob.noise_var = p.add_noise(y1)

    unused = p.genYX(nbatches) # for legacy reasons -- want to compare against a previous run
    (prob.yval, prob.xval) = p.genYX(nbatches)
    (prob.yinit, prob.xinit) = p.genYX(nbatches)
    import multiprocessing as mp
    prob.nsubprocs = mp.cpu_count()
    return prob
