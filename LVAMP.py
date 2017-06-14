#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file serves as an example of how to
a) select a problem to be solved
b) select a network type
c) train the network to minimize recovery MSE

"""
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

# Create the basic problem structure.
prob = problems.bernoulli_gaussian_trial(kappa=None,M=250,N=500,L=1000,pnz=.1,SNR=40) #a Bernoulli-Gaussian x, noisily observed through a random matrix
#prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

# build an LVAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = networks.build_LVAMP(prob,T=6,shrink='bg')
#layers = networks.build_LVAMP_dense(prob,T=3,shrink='pwgrid')

# plan the learning
training_stages = train.setup_training(layers,prob,trinit=1e-4,refinements=(.5,.1,.01))

# do the learning (takes a while)
sess = train.do_training(training_stages,prob,'LVAMP_bg_giid.npz')
