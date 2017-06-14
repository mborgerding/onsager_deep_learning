#!/usr/bin/python
from tools import problems

import numpy as np
import numpy.linalg as la
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

from scipy.io import savemat

def save_problem(base,prob):
    print('saving {b}.mat,{b}.npz norm(x)={x:.7f} norm(y)={y:.7f}'.format(b=base,x=la.norm(prob.xval), y=la.norm(prob.yval) ) )
    D=dict(A=prob.A,x=prob.xval,y=prob.yval)
    np.savez( base + '.npz', **D)
    savemat(base + '.mat',D,oned_as='column')


save_problem('problem_Giid',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=None,SNR=40))
save_problem('problem_k15',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=15,SNR=40))
save_problem('problem_k100',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=100,SNR=40))
save_problem('problem_rap1',problems.random_access_problem(1))
save_problem('problem_rap2',problems.random_access_problem(2))



