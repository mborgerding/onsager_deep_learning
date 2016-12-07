#!/usr/bin/python -u
import tensorflow as tf
import numpy as np
import utils as ut

configurations = (
        ( {} , 'problem_Giid') ,
        ( {'kappa':15.} , 'problem_k15'),
        )

for kw,base in configurations:
    # set up the inverse problem basic structure with the configuration-specific kw
    st = ut.Setup(**kw) 
    y_ph = st.get_input_node()

    # the structure inside of Setup doesn't create all the fields until `set_output_node`
    # so set a dummy value ( not really used ) 
    st.set_output_node( tf.Variable(1.0,name='junk',dtype=tf.float32) *tf.matmul( ut.adjoint(st.Psitf) , y_ph) )

    # a dictionary with the problem setup
    prob = st.get_problem_vars()

    # save as a numpy archive
    np.savez( base  + '.npz' ,**prob)
    print 'saved numpy archive %s.npz' % base

    try:
        from scipy.io import savemat
        savemat(base + '.mat',prob,oned_as='column')
        print 'saved matlab file %s.mat' % base
    except ImportError:
        pass
