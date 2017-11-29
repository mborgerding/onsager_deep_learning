#!/usr/bin/python
from __future__ import division
from __future__ import print_function
"""
This file reproduces the "lvamp-pwlin" trace from Fig 16 of

[1] Borgerding, Mark, Philip Schniter, and Sundeep Rangan. "AMP-Inspired Deep Networks for Sparse Linear Inverse Problems." IEEE Transactions on Signal Processing (2017).
Available at http://www2.ece.ohio-state.edu/~schniter/pdf/tsp17_lamp.pdf

"""
nlayers=6
trial_name = 'LVAMP_mmimo'

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

# Create the basic problem structure.
print('Creating the basic problem structure')
prob = problems.random_access_problem(2) # 1 or 2 for compressive random access or massive MIMO

# build an LVAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
print('building the LVAMP network in tensorflow')
layers = networks.build_LVAMP_dense(prob,T=nlayers,shrink='pwgrid')

# plan the learning
print('setup_training')
training_stages = train.setup_training(layers,prob,trinit=1e-4,refinements=(.5,.1,.01))

# do the learning (takes a while)
print('do_training')
sess = train.do_training(training_stages,prob,trial_name + '.npz')
