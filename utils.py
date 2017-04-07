#!/usr/bin/python
import numpy as np
import numpy.linalg as la
import tensorflow as tf
import math
import sys
import time
log10 = math.log10

__doc__ = """
utilities for inverse problems

The inference problem is, for a given input vector y in R^m,
to find the optimal sparse code vector x* in R^n that
minimizes an energy function that combines the square
reconstruction error and a sparsity penalty on the
code:
    E(y,x) = .5 || y - Psi* x ||^2 + regularize(x)
where Psi is an mxn measurement matrix whose columns
are the normalized basis vectors,

Reference:
 [1]  Gregor & LeCun "Learning Fast Approximations of Sparse Coding", ICML 2010
           http://www.cs.nyu.edu/~kgregor/gregor-icml-10.pdf
"""

class Setup:
    """
        The Setup class gives the skeletal structure of an inference algorithm
        to be learned with TensorFlow.  Setup is used both as a verb and a
        noun in this respect.
        As a verb, Setup performs functionality common to algorithms such as
            *) reading common flags
            *) initializing the sparse reconstruction problem of inferring
                Bernoulli-Gaussian x from y=Psi*x + w
            *) initializing TensorFlow session
            *) loading/saving learned variables
            *) running the Optimizer
        As a noun, Setup represents state of all variables and generators in either
        numpy form or tensorflow form.
    """
    def __init__(self,**kwargs):
        'construct the object, read common flags (e.g. dimensions,snr,...) '
        import argparse
        tf.reset_default_graph()
        tf.app.flags.FLAGS = tf.flags._FlagValues()
        tf.app.flags._global_parser = argparse.ArgumentParser()

        self.kwinit = kwargs
        self.flags = tf.app.flags
        self.flag('m', 250, 'length of observed vector y')
        self.flag(
            'n', 500, 'number of elements in synthesis dictionary ( overcomplete/compressive when n>m )')
        self.flag('mbs', 1000, 'batch size')
        self.flag('T', 10, 'number of layers')
        self.flag(
            'pnz', .1, 'probability of a synthesis element being active')
        self.flag('vnz', 1, 'variance of active synthesis elements')
        self.flag('maxiterations', 99999999, 'number of training iterations')
        self.flag('reportEvery', 10, 'report after this many steps')
        self.flag(
            'stopAfter', 50, 'halt if no progress made on the test set in this many reports/checks')
        self.flag('seed', 1, 'seed the rng (integer)')
        self.flag(
            'refinements', 1, 'after stopping, decrease trainRate,increase stopAfter and refine')
        self.flag('load', '', 'load state from a .npz file')
        self.flag('save', '', 'save state to a .npz file')
        self.flag(
            'state', '', 'equivalent to setting --load and --save to the same file')
        self.flag('snr', 40, 'SNR (dB)')
        self.flag('trainRate', 1e-4, 'training rate')
        self.flag('cpx', False, 'complex')
        self.flag('saveProblem', '', 'write the problem matrices,vectors to a file')
        self.flag('summary','','write the status to a file accessible via a Bourne-compatible shell')
        self.flag('reportOnly','','just load the state file and print out a performance report')
        self.flag('kappa',0,'if nonzero, force to measurement matrix to have the specified condition number')
        self.flag('overtrain',False,'reuse the testing (x,y) training set FOR EVERY ITERATION')
        self.flag('freeze','','do not learn the values')
        self.flag('dev','','tensorflow device')
        self.cfg = None
        self.y = None
        self.xhat = None
        self.ctr = 0  # iteration counter
        self.nictr = 0  # non-improvement counter
        self.history={}
        self.addloss=None

    def flag(self,name,dval,helpText):
        """set up a flag and its default value"""
        td = type(dval)
        if self.kwinit.has_key(name):
            dval = td(self.kwinit[name])
        f = self.flags
        funmap = {bool:f.DEFINE_boolean,int:f.DEFINE_integer,float:f.DEFINE_float,str:f.DEFINE_string}
        funmap[td](name,dval,helpText)

    def __str__(self):
        'returns a text description of the problem called from str(.) or print'
        if self.cfg is None:
            return 'run get_input_node() to perform setup'
        return 'Setup(%s)' % ','.join([('%s=%s' % (k, repr(v) )) for (k, v) in self.cfg.__dict__['__flags'].items()])

    def __getattr__(self, name):
        """returns attributes related to the command line flags
        e.g.
            setup.trainRate
        """
        return getattr(self.cfg, name)

    def get_input_node(self):
        'loads state, sets up CS problem, returns the tensorflow placeholder for y=Psi*x + w'
        if self.y is not None:
            return self.y  # already called

        cfg = self.flags.FLAGS
        self.cfg = cfg
        self.cfg._parse_flags()

        if cfg.state:
            cfg.save = cfg.state
            cfg.load = cfg.state
        if cfg.save:
            cfg.save = ensure_suffix(cfg.save, '.npz')
        if cfg.load:
            cfg.load = ensure_suffix(cfg.load, '.npz')
        self.frozen = cfg.freeze.split(',')

        if cfg.dev:
            print 'configuring for tensorflow device %s' % cfg.dev
            self.device = tf.device(cfg.dev)
            self.device.__enter__()

        self._load_state()

        if cfg.cpx:
            tf_dtype = tf.complex64
        else:
            tf_dtype = tf.float32

        # create the sensing matrix (as numpy array)
        # with unity columns (on average)
        if cfg.cpx:
            PsiScale = 1.0 / math.sqrt(2*cfg.m)
        else:
            PsiScale = 1.0 / math.sqrt(cfg.m)
        self.Psi = randn((cfg.m, cfg.n), dtype=tf_dtype.as_numpy_dtype,
                         scale=PsiScale)

        if cfg.kappa:
            U,S,V = la.svd(self.Psi,full_matrices=False)
            beta = math.exp(math.log(cfg.kappa)/(1-cfg.m))
            s = beta**np.arange(cfg.m)
            s=math.sqrt(cfg.n)*s/la.norm(s)
            self.Psi = np.dot(diagMult(U,s),V).astype(np.float32)
            assert abs(la.norm(self.Psi)**2/cfg.n -1 ) < 1e-4
            print 'measurement matrix condition number=%.4f' % la.cond(self.Psi)

        self.Psitf = tf.constant(self.Psi,dtype=tf_dtype)

        xmbs = (cfg.n, cfg.mbs)  # mini-batch generating vector x shape

        # Create a (tf) generator for a Bernoulli-Gaussian matrix (x)
        # and one for the resulting linear transform y=Psi*x .

        bernoulli = tf.to_float(tf.random_uniform(xmbs) < cfg.pnz)
        if cfg.cpx:
            self.x = tf.complex( bernoulli * randn(xmbs,tf.float32), bernoulli * randn(xmbs,tf.float32) )
        else:
            self.x = bernoulli * randn(xmbs,tf.float32)

        # Define (tf) placeholders for the desired input:output tensors.
        y_clean = tf.matmul( self.Psitf, self.x)
        expect_var_y = cfg.pnz * cfg.n / cfg.m
        self.noise_var = expect_var_y * math.pow(10, -cfg.snr / 10)

        # define (tf) observed vector
        self.y = y_clean + randn( (cfg.m, cfg.mbs), dtype=tf_dtype, \
                scale=math.sqrt( self.noise_var ) )

        self.nmse_check =  self.nmse( y_clean,self.y)

        # tensorflow generators for (y,x) pairs
        self.generators = (self.y, self.x)

        # Define placeholders for the desired input:output tensors.
        self.y_ph = tf.placeholder(tf_dtype, shape=(cfg.m, cfg.mbs))
        self.xdes_ph = tf.placeholder(tf_dtype, shape=(cfg.n, cfg.mbs))
        self.variables = {}
        self.saveconstants = {}
        self.splitcpx = {}

        return self.y_ph

    def variable(self, name, numpyVar,**kwargs):
        """create a tensorflow variable from
        the saved state if it contains _name_
        otherwise from _numpyVar_
        """
        checkload = kwargs.get('checkload', broadcast_x_like_y)
        #if iscomplex(numpyVar):
        #   if self.loaded_state.has_key(name):
        #       bx = checkload(self.loaded_state[name],numpyVar,name=name)
        #   else:
        #       bx = numpyVar
        #   tfVarRe = tf.Variable(np.real(bx) , name=name+'_re')
        #   tfVarIm = tf.Variable(np.imag(bx) , name=name+'_im')
        #   tfVar = tf.complex(tfVarRe,tfVarIm)
        #   self.splitcpx[tfVar] = (tfVarRe,tfVarIm)

        if self.loaded_state.has_key(name):
            print 'initalizing %s from %s ' % (name, self.load)
            numpyVar = checkload(self.loaded_state[name],numpyVar,name=name)
        else:
            print 'default initialization for %s' % (name)

        assert not iscomplex(numpyVar),'TODO'

        if name in self.frozen:
            print 'FROZEN AS CONSTANT:%s' % name
            tfVar = tf.constant( numpyVar )
            self.saveconstants[name] = numpyVar
        else:
            tfVar = tf.Variable(  numpyVar )
            self.variables[name] = tfVar
        return tfVar

    def penalize_if_less_than(self,tfvar,minValue=1e-9):
        if len(tfvar.get_shape()):
            tfvar = tf.reduce_min(tfvar)
        tfvar = tf.minimum( tfvar-minValue ,0)
        tfvar = tfvar * tfvar
        self.add_penalty( tfvar)

    def add_penalty(self,lossval):
        'add a given tensorflow scalar to the loss function'
        if self.addloss is None:
            self.addloss = lossval
        else:
            self.addloss = self.addloss + lossval

    def halfFrobeniusNormSquared(self,x):
        if iscomplex(x):
            return self.halfFrobeniusNormSquared(tf.real(x)) + self.halfFrobeniusNormSquared(tf.imag(x))
        else:
            return tf.nn.l2_loss(x)

    def nmse(self, xtrue, xhat):
        return self.halfFrobeniusNormSquared(xtrue-xhat) / self.halfFrobeniusNormSquared(xtrue)

    def noisy_Psi(self):
        'create a coarse estimate of Psi from a randomly drawn (x,y) pair'
        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer())
            (y, x) = sess.run(self.generators)
            #return adjoint(la.lstsq(adjoint(x),adjoint(y)))
        return np.matmul(y ,la.pinv(x) )


    def set_output_node(self, xhat,**extra_vars):
        """provide the Setup with the output estimator from the CS inference
        algorithm
        """
        assert self.xhat is None
        self.xhat = xhat
        self.loss = self.nmse(self.xdes_ph,self.xhat)
        objective_func = self.loss
        if self.addloss is not None:
            objective_func = objective_func + self.addloss

        # by keeping the training_rate in tensor, we can modify the
        # training_rate in response to training effectiveness
        self.tr = tf.Variable(self.cfg.trainRate)
        self.train = tf.train.AdamOptimizer(self.tr).minimize( objective_func )
        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init)

        # create two datasets held back from training
        # test:  for making decisions about stepsizes etc.
        # validation: ONLY for error reporting
        (ytst, xtst) = self.sess.run(self.generators)
        (yval, xval) = self.sess.run(self.generators)
        self.feed_dict_test = {self.y_ph: ytst, self.xdes_ph: xtst}
        self.feed_dict_val = {self.y_ph: yval, self.xdes_ph: xval}

        self.yval = yval
        self.xval = xval
        if self.cfg.reportOnly:
            self._reportAndExit()

        v=dict(extra_vars)
        v['y']=ytst
        v['x']=xtst
        v['yval']=yval
        v['xval']=xval
        v['A']=self.Psi
        self._save_problem(**v)

    def get_problem_vars(self,**extra_vars):
        'return a dictionary of numpy objects describing the test setup'
        v = dict(extra_vars) # copy
        v['args'] = str(self)
        v['y']=self.feed_dict_test[self.y_ph]
        v['x']=self.feed_dict_test[self.xdes_ph]
        v['yval']=self.feed_dict_val[self.y_ph]
        v['xval']=self.feed_dict_val[self.xdes_ph]
        v['A']=self.Psi
        return v


    def current_value(self,tfVar):
        return self.sess.run( tfVar,feed_dict=self.feed_dict_test)

    def run(self,pfunc=None):
        """Perform the learning, with optional refinement steps
        """
        for r in range(max(0,self.cfg.refinements)+1):
            if r:
                self._next_stage()
            for i in range(self.cfg.maxiterations/self.cfg.reportEvery ):
                if not self.step():
                    break
                if pfunc is not None:
                    retval = pfunc()
                    # if the pfunc returns anything, return it if it evaluates to false
                    if retval is not None and not retval:
                        return retval

    def _next_stage(self):

        self.cfg.stopAfter *= 2
        self.cfg.trainRate *= .1
        self.sess.run(self.tr.assign(self.cfg.trainRate))
        print 'refining ... trainRate=%s stopAfter=%d' % (self.cfg.trainRate, self.cfg.stopAfter)
        self.nictr = 0

    def _reportAndExit(self):
        assert(self.cfg.load);
        sr = self.sess.run

        fn = self.cfg.reportOnly
        f = sys.stdout if fn=='-' else open(fn,'w')
        f.write( 'nmse_val=%s\n' % (10 * log10(sr(self.loss, feed_dict=self.feed_dict_val)) ) )
        f.write( 'nmse_test=%s\n' % ( 10 * log10(sr( self.loss, feed_dict=self.feed_dict_test)) ) )
        xhat = sr( self.xhat, feed_dict=self.feed_dict_val )
        xhat_db1 = xhat_debias_lsq(xhat,self.Psi,self.yval)
        f.write( 'nmse_db1=%s\n' % ( 20*log10( la.norm(xhat_db1-self.xval)/la.norm(self.xval) ) ) )
        xhat_db2 = xhat_debias_lmmse(xhat,self.Psi,self.yval,self.noise_var)
        f.write( 'nmse_db2=%s\n' % ( 20*log10( la.norm(xhat_db2-self.xval)/la.norm(self.xval) ) ) )
        sys.exit(0)

    def genyx(self):
        cfg = self.cfg;
        X = (np.random.random_sample(size=(self.n,self.mbs)) < self.pnz) * np.random.normal(size=(self.n,self.mbs))
        #AX = np.matmul(self.Psi,X)
        Y = np.matmul(self.Psi,X)  + np.random.normal( size=(self.m,self.mbs), scale=math.sqrt( self.noise_var ) )
        """
        y_clean = tf.matmul( self.Psitf, self.x)
        expect_var_y = cfg.pnz * cfg.n / cfg.m
        self.noise_var = expect_var_y * math.pow(10, -cfg.snr / 10)

        # define (tf) observed vector
        self.y = y_clean + randn( (cfg.m, cfg.mbs), dtype=tf_dtype, \
        scale=math.sqrt( self.noise_var ) )
        """
        return Y,X


    def step(self):
        sr = self.sess.run
        'perform a set of optimization steps'
        # report initial NMSE for test set ONLY on the initial run
        if self.ctr == 0:
            # initial step
            self.tstart = time.time()
            self.nmse_test = 10 * log10(sr( self.loss, feed_dict=self.feed_dict_test))
            self.nmse_first = self.nmse_test
            self.nmse_best = self.nmse_test
            self._stash_variables()
            self.nmse_val0 = 10 * log10(sr(self.loss, feed_dict=self.feed_dict_val))
            print 'initial test_nmse=%.3fdB val_nmse=%.3fdB' % ( self.nmse_test , self.nmse_val0)


        for i in xrange(self.cfg.reportEvery):
            if self.cfg.overtrain:
                fd_trn = self.feed_dict_test
            else:
                (ytrn, xtrn) = sr(self.generators)
                #(ytrn, xtrn) = self.genyx()
                # inputs and outputs for training
                fd_trn = {self.y_ph: ytrn, self.xdes_ph: xtrn}
            # perform fprop,bprop on the training set fd_trn
            sr(self.train, feed_dict=fd_trn)

        self.nmse_test = 10 * log10(sr(self.loss, feed_dict=self.feed_dict_test))
        self.nmse_val = 10 * log10(sr(self.loss, feed_dict=self.feed_dict_val))

        self.ctr += self.cfg.reportEvery

        if math.isnan( self.nmse_test + self.nmse_val ):
            print 'giving up as hopeless NMSE=nan' 
            for (k, v) in self.stashed.items():
                print '%s=%s' %(repr(k),repr(v))
            self._unstash_variables()
            self.sess.run(self.init)
            return False

        if self.nmse_test >= self.nmse_best:
            self.nictr += 1
            if self.nictr >= self.cfg.stopAfter:
                print 'no improvement in %d checks,Test NMSE=%.3fdB' % (self.nictr, self.nmse_test)
                if self.nmse_best == self.nmse_first:
                    print 'never improved -- reverting'
                    # if none of the training at this trainRate improved
                    # anything, reset
                    self._unstash_variables()
                return False
            if self.nmse_test >= self.nmse_best + 3:
                print 'giving up as hopeless NMSE=%.3fdB' % self.nmse_test
                self._unstash_variables()
                return False
        else:
            self.nmse_best = self.nmse_test
            self._stash_variables()
            self.nictr = 0  # reset

        def h(k,v):
            if not self.history.has_key(k):
                self.history[k] = []
            self.history[k].append( v )
            print '%s=%s' % (k,str(v)) ,

        h('step',self.ctr)
        h('elapsed',(time.time() - self.tstart) )
        h('nic',self.nictr )
        h('nmse_test',self.nmse_test)
        h('nmse_val',self.nmse_val)
        h('nmse_val0',self.nmse_val0)
        print ''
        self._save_state()
        return True

    def _stash_variables(self):
        self.stashed = {}
        for (k, v) in self.variables.items():
            self.stashed[k] = self.sess.run(v, feed_dict=self.feed_dict_test)

    def _unstash_variables(self):
        for (k, v) in self.variables.items():
            self.sess.run(v.assign(self.stashed[k]))
        self._save_state()

    def describe(self, tfvars):
        desc = ''
        for tfv in tfvars:
            desc += tfv.name
            val = self.sess.run(tfv)
            desc += str(val)
            desc += ' '
        return desc

    def _save_problem(self,**kwargs):
        if self.cfg.saveProblem:
            np.savez(self.cfg.saveProblem,**kwargs)
            sys.exit(0)

    def _save_state(self):
        if self.cfg.save:
            STATE = dict(self.loaded_state)  # make a copy
            for (k,v) in self.history.items():
                STATE[k] = v[-1]
            for (k, v) in self.variables.items():
                STATE[k] = self.sess.run(v, feed_dict=self.feed_dict_test)
                # print 'saving %s (%s) ' % (k,v.name)
            for (k, v) in self.saveconstants.items():
                STATE[k] = v
            np.savez(self.cfg.save, **STATE)
        if self.cfg.summary:
            f=open(self.cfg.summary,'w')
            f.write(''.join([ '%s=%s\n' %(k,v[-1]) for (k, v) in self.history.items()]) )


    def _load_state(self):
        cfg = self.cfg
        self.loaded_state = {}
        if cfg.load:
            try:
                self.loaded_state = dict(np.load(cfg.load).items())
                # print 'loaded:' + str(self.loaded_state.keys())
            except IOError as e:
                sys.stderr.write(str(e) + '\n')

        if cfg.seed != 0:
            np.random.seed(cfg.seed)
            tf.set_random_seed(cfg.seed)


def colnorm(y, pnorm=2):
    if iscomplex(y):
        yr = tf.real(y)
        yi = tf.imag(y)
        m2 = yr*yr + yi*yi
    else:
        m2 = y*y

    outshape = (1, int(y.get_shape()[1]))
    if pnorm == 2:
        y = tf.sqrt(tf.reduce_sum(m2, 0))
    elif pnorm == 0:
        y = tf.reduce_sum( tf.to_float(m2>0),0)
    return tf.reshape(y, outshape)

def ensure_suffix(s, sfx):
    if len(s) < len(sfx) or s[-len(sfx):] != sfx:
        return s + sfx
    else:
        return s


def randn(shape, dtype=np.float32, scale=1):
    if dtype is tf.complex64:
        return tf.complex( tf.random_normal(shape,stddev=scale) ,
                           tf.random_normal(shape,stddev=scale) )
    elif dtype is tf.float32:
        return tf.random_normal(shape,stddev=scale)
    elif dtype is np.complex64:
        return randn(shape, np.float32, scale) + 1j * randn(shape, np.float32, scale)
    else:
        return np.random.normal(size=shape, scale=scale).astype(dtype)

def same(x, y, reltol=1e-6):
    return la.norm(x - y) <= reltol * max(la.norm(x), la.norm(y))

def iscomplex(X):
    if hasattr(X,'dtype'):
        dt = X.dtype
        if hasattr(dt,'as_numpy_dtype'):
            dt = dt.as_numpy_dtype
        return np.dtype(dt) is np.dtype(np.complex64)
    else:
        return type(X) is complex

def conj(X):
    if hasattr(X,'conjugate'):
        return X.conjugate()
    if hasattr(X,'conj'):
        return X.conj()
    if hasattr(X,'dtype') and X.dtype is tf.complex64:
            return tf.conj(X)
    return X

def adjoint(X):
    if hasattr(X,'T'):
        Xt = X.T
    else:
        Xt = tf.transpose(X)
    if iscomplex(Xt):
        return conj(Xt)
    else:
        return Xt

def inner_product(X, Y):
    return tf.reduce_sum(tf.mul(conj(X), Y))


def conv2d(X, F, adjoint=False, func=tf.nn.conv2d):
    """
    Y = conv2d(X,F,adjoint=False)
    Perform 2d convolution of a (possibly complex) input image X with
    a filter set in F.
    X has shape (batch, x_height, x_width, x_channels)
    if func is tf.nn.conv2d:
      F has shape (filter_height, filter_width, x_channels, y_channels)
      Y has shape (batch, y_height, y_width, y_channels)
      output[b, i, j, k] =
        sum_{di, dj, q} input[b, i + di, j + dj, q] *
                        filter[di, dj, q, k]
    if func is tf.nn.depthwise_conv2d:
      F has shape (filter_height, filter_width, x_channels, channel_multiplier)
      Y has shape (batch, y_height, y_width, x_channels * channel_multiplier)
      output[b, i, j, k * channel_multiplier + q] =
          sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                       filter[di, dj, k, q]

    Known limitation: adjoint=True and func=tf.nn.depthwise_conv2d can only be used
    together when channel_multiplier==1.

    For more info, see tensorflow.nn.conv2d and tensorflow.nn.depthwise_conv2d
    """
    re = tf.real
    im = tf.imag
    cpx = tf.complex
    if adjoint:
        if func is not tf.nn.conv2d and F.get_shape()[3] != 1:
            raise Exception(
                'cannot yet do adjoint for func other than nn.conv2d ')

        if func is tf.nn.conv2d:
            F = tf.transpose(F, (0, 1, 3, 2))

        def rev(X): return tf.reverse(X, [True, True, False, False])
        if F.dtype is tf.complex64:
            F = cpx(rev(re(F)), tf.neg(rev(im(F))))
        else:
            F = rev(F)

    if X.dtype is tf.complex64 and F.dtype is tf.complex64:
        return cpx(conv2d(re(X), re(F), func=func) - conv2d(im(X), im(F), func=func),
                   conv2d(re(X), im(F), func=func) + conv2d(im(X), re(F), func=func))
    elif X.dtype is tf.complex64:  # imag(F) is zero
        return cpx(conv2d(re(X), F, func=func), conv2d(im(X), F, func=func))
    elif F.dtype is tf.complex64:  # imag(X) is zero
        return cpx(conv2d(X, re(F), func=func), conv2d(X, im(F), func=func))
    else:
        return func(X, F, [1, 1, 1, 1], 'SAME')


def softThreshold(x, lam):
    "implement a soft threshold function y=sign(x)*max(0,abs(x)-lam)"
    lam = tf.maximum(lam, 0)
    if iscomplex(x):
        xr = tf.real(x)
        xi = tf.imag(x)
        xmag = tf.sqrt(xr*xr + xi*xi)
        scale = tf.ones(x.get_shape() ,dtype=tf.float32  ) - lam/xmag
        scale = tf.maximum(
                    scale,
                    tf.zeros(x.get_shape(), dtype=tf.float32  ) )
        return tf.complex( scale*tf.real(x), scale*tf.imag(x))
    else:
        return tf.sign(x) * tf.maximum(tf.abs(x) - lam, 0)

def bgest(r,rvar,omega,beta):
    r2 = r*r
    rho = tf.exp(omega - beta*r2*.5/rvar)
    xhat = r*beta/(1+rho)
    dxdr = beta * (1+rho*(1+beta*r2/rvar) ) / tf.square( 1.0 + rho )
    return (xhat,dxdr)

def pwlin(r,rvar,theta):
    """Implement the piecewise linear shrinkage function.
        With minor modifications and variance normalization.
        theta[0] : abscissa of first vertex, scaled by sqrt(rvar)
        theta[1] : abscissa of second vertex, scaled by sqrt(rvar)
        theta[2] : slope from origin to first vertex
        theta[3] : slope from first vertex to second vertex
        theta[4] : slope after second vertex
    """
    vtx = theta[0:2]
    slopes = theta[2:5]

    # scale each column by sqrt(rvar)
    scale_out = tf.sqrt(rvar)
    scale_in = 1/scale_out
    rs = tf.sign(r*scale_in)
    ra = tf.abs(r*scale_in)

    # split the piecewise linear function into regions
    rgn0 = tf.to_float( ra<vtx[0])
    rgn1 = tf.to_float( ra<vtx[1]) - rgn0
    rgn2 = tf.to_float( ra>=vtx[1])
    xhat = scale_out * rs*(
            rgn0*slopes[0]*ra +
            rgn1*(slopes[1]*(ra - vtx[0]) + slopes[0]*vtx[0] ) +
            rgn2*(slopes[2]*(ra - vtx[1]) +  slopes[0]*vtx[0] + slopes[1]*(vtx[1]-vtx[0]) )
            )
    dxdr =  slopes[0]*rgn0 + slopes[1]*rgn1 + slopes[2]*rgn2
    return (xhat,dxdr)

def exp_shrinkage(r,rvar,theta):
    r2 = tf.square(r)
    den = -1/(2*theta[0]*tf.sqrt(rvar) )
    rho = tf.exp( r2 * den)
    xhat = r*( theta[1] + theta[2] * rho )
    dxdr = theta[1] + theta[2] * rho*(2*r2*den + 1 )
    return (xhat,dxdr)

def spline_shrinkage(r,rvar,theta):
    scale = theta[0]*tf.sqrt(rvar)
    rs = tf.sign(r)
    ar = tf.abs(r/scale);
    ar2 = tf.square(ar)
    ar3 = ar*ar2
    reg1 = tf.to_float(ar<1)
    reg2 = tf.to_float(ar<2)-reg1
    ar_m2 = 2-ar
    ar_m2_p2 = tf.square(ar_m2)
    ar_m2_p3 = ar_m2*ar_m2_p2
    beta3 = ( (2./3 - ar2  + .5*ar3)*reg1 + (1./6*(ar_m2_p3))*reg2 )
    xhat = r*(theta[1] + theta[2]*beta3)
    dxdr = tf.gradients(xhat,r)[0]
    return (xhat,dxdr)


def eta(rhat,lam):
    'soft threshold eta(r,lam) = sign(r)*max(0,abs(r)-lam)'
    return  rhat - np.clip(rhat,-lam,lam)

def etaopt(rhat,rvar,x,**kwargs):

    avec = np.linspace(.0001,4,25)
    for k in range(9):
        da = avec[1] - avec[0]

        err = np.ones_like(avec)
        for k in range(len(avec)):
            lam = avec[k] * np.sqrt(rvar)
            xhat0 = eta(rhat,lam)
            beta = kwargs.get('beta', (xhat0*x).sum()/(1e-12+ la.norm(xhat0)**2)) # for a given lambda,optimal beta is available in a closed-form solution
            xhat = beta*xhat0
            err[k] = la.norm(xhat - x)
        kmin = np.argmin(err)
        avec = np.clip(avec[kmin] + np.linspace(-1,1,25)*da,0,9999)
    alpha = avec[kmin]
    lam = alpha * np.sqrt(rvar)
    xhat0 = eta(rhat,lam)
    beta = kwargs.get('beta', (xhat0*x).sum()/la.norm(xhat0)**2)
    xhat = beta*xhat0

    return (xhat,alpha,beta)


def etaopt2(rhat,x,**kwargs):

    minlam = 0
    maxlam = np.abs(rhat).max()

    for k in range(5):
        lamvec = np.linspace(minlam,maxlam,11)
        dlam = lamvec[1] - lamvec[0]
        err = np.ones_like(lamvec)

        for i in range(len(lamvec)):
            lam = lamvec[i]
            xhat0 = eta(rhat,lam)
            beta = kwargs.get('beta', (xhat0*x).sum()/(1e-12+ la.norm(xhat0)**2)) # for a given lambda,optimal beta is available in a closed-form solution
            xhat = beta*xhat0
            err[i] = la.norm( xhat - x)
        minidx = np.argmin(err)
        bestlam = lamvec[minidx]
        minlam = max(0,bestlam - dlam)
        maxlam = bestlam + dlam
    lam = bestlam
    xhat0 = eta(rhat,lam)
    beta = kwargs.get('beta', (xhat0*x).sum()/(1e-12+ la.norm(xhat0)**2))
    xhat = beta*xhat0
    return (xhat,lam,beta)

def BernoulliGaussianEstimator(r,rvar,lam,xvar1):
    """ Estimate E[x|r] for a Bernoulli-Gaussian vector observed in iid Gaussian noise
        i.e.
        R-X ~ Normal(0,rvar)
        p_X(x) = (1-lambda) * delta(x) + lambda*Normal(x;0,xvar1)
    """

    beta = 1/(1+rvar/xvar1)
    rsq = tf.square(r)/rvar
    rho = (1/lam - 1) * tf.sqrt(1 +xvar1/rvar) * tf.exp( -.5*beta*rsq )
    x = beta * (r /(1+rho))
    dxdr = beta * (1+rho*(1+beta*rsq) ) / tf.square( 1 + rho );
    return (x,dxdr)

def matmul(X, Y):
    """"implements a broadcasting matrix multiplication Z=X*Y where the
    last two dimensions of X,Y must be compatible for matrix multiplication
    and the leading dimensions are automatically broadcast."""

    if iscomplex(X) != iscomplex(Y):
        if iscomplex(X):
            return tf.complex( tf.matmul( tf.real(X),Y),tf.matmul(tf.imag(X)),Y)
        else:
            return tf.complex( tf.matmul(X,tf.real(Y)),tf.matmul(X,tf.imag(Y)))
    else:
        return tf.matmul(X,Y)

    #shX = X.get_shape().as_list()
    #shY = Y.get_shape().as_list()
    #if shX[0:-2] == shY[0:-2]:
    #    # initial
    #    return tf.batch_matmul(X, Y)
    #elif len(shY) == 2:
    #    shZ = shX[0:-1] + shY[-1:]
    #    return tf.reshape(tf.matmul(tf.reshape(X, [-1, shX[-1]]), Y), shZ)
    #elif len(shX) == 2:
    #    shZ = shY[0:-2] + shX[-2:-1] + shY[-1:]
    #    # swap the last and second-to-last dimensions of the operands and
    #    # recurse
    #    dimTran = range(len(shY))
    #    dimTran[-1] = dimTran[-1] - 1
    #    dimTran[-2] = dimTran[-2] + 1
    #    tmp1 = matmul(tf.transpose(Y, dimTran),  tf.transpose(X))
    #    shZtran = shZ[:]
    #    shZtran[-1] = shZ[-2]
    #    shZtran[-2] = shZ[-1]
    #    tmp2 = tf.reshape(tmp1, shZtran)
    #    return tf.transpose(tmp2, dimTran)


def gen_bern(shape, prob_non_zero, dtype=tf.float32):
    """  Create a tensorflow generator for a Bernoulli tensor
    with the given shape, P(nonzero) and data type.
    """
    result = tf.to_float(tf.random_uniform(shape) < prob_non_zero)
    if dtype is tf.complex64:
        result = tf.complex(result, tf.zeros_like(result))
    return result


def gen_gauss(shape, stddev, dtype=tf.float32):
    """ Create a tensorflow generator for a Gaussian
    (or circular Gaussian tensor) with the given
    """
    if dtype is tf.complex64:
        return tf.complex(gen_gauss(shape, stddev, tf.float32),
                          gen_gauss(shape, stddev, tf.float32))
    else:
        return tf.random_normal(shape, stddev=stddev, dtype=dtype)


def fillout(x, n):
    """ y=fillout(x,n)
      where x,y are numpy arrays
      len(y) is always n
      if len(x) >= n, y=x[0:n]
      if len(x) <= n, y=[x x[-1] ...x[-1] ]  (i.e. repeat last element until len(y)==n )
      """
    if len(x) >= n:
        return x[0:n]
    else:
        # repeat in the zero dimension
        repdims=np.ones_like(x.shape)
        repdims[0] = n - len(x)
        y = np.append(x, np.tile( x[-1] ,repdims) ,0)
        return y

def broadcast_x_like_y(x,y,**kwargs):
    xs = x.shape
    if hasattr(y,'shape'):
        ys = y.shape
    else:
        ys=()
    z = np.empty(ys)
    if kwargs.has_key('name'):
        prefix = kwargs['name'] + ':'
    else:
        prefix = ''

    if len(xs) < len(ys):
        if xs == ys[1:]:
            print '%srepeating data to make %s like %s' % (prefix, str(xs),str(ys) )

            x = np.reshape(x,[1]+list(x.shape))
            z = np.tile(x,[ys[0]] + list(np.ones_like(xs)) )
        else:
            raise "not yet"
    elif xs==ys:
        # already the right size

        z = x
    else:
        if xs[1:] == ys[1:]:
            if ys[0] > xs[0]:
                print '%sappending to make shape %s like %s' % (prefix, str(xs),str(ys) )
                # repeat last element
                repdims=np.ones_like(xs)
                repdims[0] = ys[0] - xs[0]
                z = np.append(x, np.tile( x[-1] ,repdims) ,0)
            else:
                print '%struncating to make shape %s like %s' % (prefix, str(xs),str(ys) )
                # truncate
                z = x[0:ys[0]]
        elif xs[:-1] == ys[:-1] and xs[-1] < ys[-1]:
            print '%srepeating the last dimension to make shape %s like %s' % (prefix, str(xs),str(ys) )
            assert( len(xs)==2)
            repdims=np.ones_like(xs)
            repdims[-1] = ys[-1] - xs[-1]
            xlast= np.reshape(x[:,-1],(xs[0],1))
            z = np.append(x, np.tile( xlast ,repdims) ,1)
        else:
            print 'xs=%s' % repr(xs)
            print 'ys=%s' % repr(ys)
            raise '%s nope, not yet' % prefix
    return z

def xhat_debias_lmmse(xhat,A,y,noise_var):
    """ return a LMMSE debiased version of xhat.
    i.e.  perform LMMSE xhat = Cxx*B' * ( ( B*Cxx*B' + noise_var*I ) \ y)
    (assumes zero means)
    where B is only the subset of A columns corresponding to nonzero xhat.
    If xhat and y are matrices (multi-measurement vectors) each column is debiased
    independently.
    """
    (m,n) = A.shape
    q = xhat.shape[1]
    xhat_db = np.zeros_like(xhat);
    for i in range(q):
        supp = np.nonzero(xhat[:,i])[0]
        B = A[:,supp]
        Cxx = np.identity( supp.shape[0],dtype=B.dtype)
        TMP1 = np.matmul( Cxx ,adjoint(B) )
        TMP2 = np.matmul(B,TMP1) + noise_var * np.identity(m,dtype=B.dtype)
        xhat_db[supp,i] = np.matmul( TMP1 , la.solve( TMP2,y[:,i]) )
    return xhat_db

def xhat_debias_lsq(xhat,A,y):
    """ return a least-squares debiased version of xhat.
    i.e.  xhat_lsq = B \ y
    where B is only the subset of A columns corresponding to nonzero xhat.
    If xhat and y are matrices (multi-measurement vectors) each column is debiased
    independently.
    """
    (m,n) = A.shape
    q = xhat.shape[1]
    xhat_db = np.zeros_like(xhat);
    for i in range(q):
        supp = np.nonzero(xhat[:,i])[0]
        B = A[:,supp]
        xhat_db[supp,i] = la.lstsq(B,y[:,i])[0]
    return xhat_db

def diagMult(A, B):
    """Multiply a full matrix by a diagonal matrix.  One of the arguments must be a 1d array.
    (based on idea from Pietro Berkes on scipy mailing list)
        diagMult(d,A) == dot(diag(d), A)
        diagMult(A,d) == dot(A, diag(d))
    """
    if hasattr(A,'get_shape') and len(A.get_shape())==1: # left multiplying by diagonal matrix
        return tf.transpose(A*tf.transpose(B))
    if hasattr(A,'shape') and len(A.shape)==1: # left multiplying by diagonal matrix
        return (A*B.T).T
    else:
        return A*B # right multiplying by diagonal matrix


def genXY(A, **kw ):
    """create (X,Y) mini-batch for A.
    keywords: M,N,L,pnz,SNR
    """
    N=kw.get('N',500)
    M=kw.get('M',250)
    L=kw.get('L',1000)
    pnz=kw.get('pnz',.1)
    snr = kw.get('SNR',40)
    expect_var_y = pnz * la.norm(A)**2 / M
    noise_var = expect_var_y * math.pow(10, -snr / 10)
    X = (np.random.random_sample(size=(N,L)) < pnz) * np.random.normal(size=(N,L))
    Y = np.matmul(A,X) + np.random.normal( size=(M,L), scale=math.sqrt( noise_var ) )
    return (X,Y)

def main():
    print 'testing pseudoinverse -> MMSE estimator'
    st = Setup(m=20,n=20,reportEvery=50,maxiterations=500,refinements=0)
    y_ph = st.get_input_node()
    print st
    PsiPinv = st.variable('G', la.pinv(st.Psi) )
    xhat = tf.matmul( PsiPinv , y_ph )
    st.set_output_node(xhat)
    st.run()


if __name__=="__main__":
    main()

