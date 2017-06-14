#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf

def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other

def setup_training(layer_info,prob, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None ):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    nmse_=[]
    trainers_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    maskX_ = getattr(prob,'maskX_',1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ *maskX_)

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    for name,xhat_,var_list in layer_info:
        loss_  = tf.nn.l2_loss( xhat_ - prob.x_)
        nmse_  = tf.nn.l2_loss( (xhat_ - prob.x_)*maskX_) / nmse_denom_
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_)
            training_stages.append( (name+' trainrate=' + str(fm) ,xhat_,loss_,nmse_,train2_,()) )
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine).minimize(loss_)
        training_stages.append( (name+' final refine ' + str(final_refine) ,xhat_,loss_,nmse_,train2_,()) )

    return training_stages


def do_training(training_stages,prob,savefile,ivl=10,maxit=1000000,better_wait=5000):
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(100*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y,prob.x_:x} )
        done = np.append(done,name)

        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)
    return sess
