#!/usr/bin/python
from __future__ import division
import numpy as np
import math
import os
import time
import numpy.linalg as la
from tfinterp import interp1d_
sqrt=np.sqrt
pi = math.pi

def hexagonal_uniform(N,as_complex=False):
    'returns uniformly distributed points of shape=(2,N) within a hexagon whose minimum radius is 1.0'
    phi = 2*pi/6 *.5
    S = np.array( [[1,1],np.tan([phi,-phi])] ) # vectors to vertices of next hexagon ( centered at (2,0) )a
    # uniformly sample the parallelogram defined by the columns of S
    v = np.matmul(S,np.random.uniform(0,1,(2,N)))
    v[0] = 1 - abs(v[0]-1) # fold back to make a triangle
    c = (v[0] + 1j*v[1]) * np.exp( 2j*pi/6*np.floor( np.random.uniform(0,6,N) ) ) # rotate to a random sextant
    if as_complex:
        return c
    else:
        return np.array( (c.real,c.imag) )

def left_least_squares(x,y,rcond=-1,fast=False):
    'find the A that best fits y-A*x'
    if fast:
        return la.lstsq( np.matmul(x,x.T) ,np.matmul(x,y.T) ,rcond=rcond )[0].T  # faster, but less stable
    else:
        return la.lstsq( x.T,y.T,rcond=rcond)[0].T

def rms(x,axis=None):
    'calculate the root-mean-square of a signal, if axis!=None, then reduction will only be along the given axis/axes'
    if np.iscomplexobj(x):
        x=abs(x)
    return np.sqrt(np.mean(np.square(x),axis) )

def nlfunc(r,sc,grid,gg,return_gradient=True):
    'returns xhat_nl = rhat_nl * interp( rhat_nl / sc,grid,gg) and optionally the gradient of xhat_nl wrt rhat_nl'
    g = r * np.interp(r/sc,grid,gg)
    if return_gradient:
        #I had some code that computed the gradient, but it was far more complicated and no faster than just computing the empirical gradient
        # technically, this computes a subgradient
        dr = sc * (grid[1]-grid[0]) * 1e-3
        dgdr = (nlfunc(r+.5*dr,sc,grid,gg,False) - nlfunc(r-.5*dr,sc,grid,gg,False)) / dr
        return (g,dgdr)
    else:
        return g


def nlfunc_(r_,sc_,grid,gg_,return_gradient=True):
    'returns xhat_nl = rhat_nl * interp( rhat_nl / sc,grid,gg) and optionally the gradient of xhat_nl wrt rhat_nl'
    g_ = r_ * interp1d_(r_/sc_,grid,gg_)
    if return_gradient:
        #I had some code that computed the gradient, but it was far more complicated and no faster than just computing the empirical gradient
        # technically, this computes a subgradient
        dr_ = sc_ * (grid[1]-grid[0]) * 1e-3
        dgdr_ = (nlfunc_(r_+.5*dr_,sc_,grid,gg_,False) - nlfunc_(r_-.5*dr_,sc_,grid,gg_,False)) / dr_
        return (g_,dgdr_)
    else:
        return g_

def crandn(shape,set_mag=None):
    'circular symmetric Gaussian with variance 2 (real,imag each being var=1) '
    X= np.random.normal( size=tuple(shape)+(2,)).view(np.complex128)[...,0]
    if set_mag is not None:
        X = X *set_mag / abs(X)
    return X

def random_qpsk( *shape):
    return ((np.random.uniform( -1,1,size=shape+(2,) ) > 0)*2-1).astype(np.float32).view(np.complex64)[...,0]

class Problem(object):

    @staticmethod
    def scenario1():
        return dict(Nr=1,C=1,Nu=512,Ns=64,beta=.01,SNR_dB=10.0,L=5,ang=10,rice_k_dB=10,ple=4,mmv2d=True,normS=1)

    @staticmethod
    def scenario2():
        return dict(Nr=64,C=7,Nu=64,Ns=64,beta=1,SNR_dB=20.0,L=5,ang=10,rice_k_dB=10,ple=4,mmv2d=True,normS=1)

    def __init__(self, Nr=64, C=7, Nu=64, Ns=64, beta=.01,L=5,ang=10,rice_k_dB=10,ple=4,SNR_dB=10.0,ambig=False,scramble=False,S=None,cpx=False,mmv2d=False,normS=None):
        """
        Nr : number of Rx antennas
        C : number of cells (>1 indicates there are "far" users)
        Nu : max # users per cell
        Ns : spreading code length
        beta :  user load (i.e.,expected active / total user ratio)
        L : paths per cluster
        ang : angular spread within cluster (in degrees)
        rice_k_dB : rice k parameter in dB
        ple : path-loss exponent: gain = 1/(1+d^ple) for distance d
        S : set of spreading codes, shape=(Ns,C*Nu) """
        if S is None:
            S = random_qpsk(Ns,C*Nu)
        self.Nr = Nr
        self.C = C
        self.Nu = Nu
        self.Ns = Ns
        self.beta = beta
        self.L = L
        self.ang = ang
        self.rice_k_dB = rice_k_dB
        self.ple = ple
        self.SNR_dB = SNR_dB
        self.ambig = ambig
        self.scramble = scramble
        self.cpx = cpx
        self.mmv2d = mmv2d

        if self.cpx == np.iscomplexobj(S):
            self.S = S
        else:
            if not self.cpx:
                top = np.concatenate( (S.real, -S.imag),axis=1 )
                btm = np.concatenate( (S.imag, S.real),axis=1 )
                self.S = np.concatenate( (top,btm),axis=0 )
            else:
                assert False,'WHY!?'

        if self.cpx:
            assert self.S.shape == (Ns,C*Nu)
        else:
            assert self.S.shape == (2*Ns,2*C*Nu)

        if normS is not None:
            dnorm = np.asarray(normS) / np.sqrt( np.square(self.S).sum(axis=0) )
            self.S = self.S * dnorm
        self.timegen = 0 # time spent waiting for generation of YX (does NOT count subprocess cpus time if nsubprocs>0


    def genX(self,batches=1):
        """generate one or more batches(i.e. random draws) of active users with Ricean channels
        batches : number of independent realizations to generate
        If cpx, the returned X has shape (batches,C*Nu,Nr),
        otherwise (batches,2*C*Nu,Nr)
        """
        Nr,C,Nu,Ns,S = self.Nr,self.C,self.Nu,self.Ns,self.S
        L,ang,rice_k_dB,ple = self.L,self.ang,self.rice_k_dB,self.ple
        X = np.zeros((batches,C,Nu,Nr),dtype=np.complex64)

        for i in range(batches):
            for c in range(C): #c==0 indicates all users are in the base stations' cell ("near")
                ###################################
                # choose how many and which users are active in this cell
                K = np.random.binomial(Nu,self.beta) # number of active users in this cell E[K] = Nu*beta
                active_users = np.random.permutation(Nu)[:K]

                #NOTE: Tensors below have shape (user,path,angle), until Z when we sum the path dimension.

                # how far (weak) is each user?
                if c==0:
                    dist = abs( hexagonal_uniform( K ,as_complex=True) )
                elif 0<c<7:
                    dist = abs( 2+hexagonal_uniform( K ,as_complex=True) ) 
                else:
                    assert False,'assuming 1 or 7 hexagonal cells'
                dist.shape = (K,1,1)
                gain = 1/(1+dist**ple)

                # The L paths per user impinge on our linear array with clustered angles theta.
                # All paths per user start at theta0 and are uniformly distributed in the next `ang` degrees.
                # (theta units=radians,zero means broadside to the linear array)
                theta0 = np.random.uniform(0,2*pi,(K,1,1))
                theta = np.mod( theta0 + np.random.uniform(0,ang*pi/180,(K,L,1)) ,2*pi)

                # different Ricean gains for each of the paths
                direct_path = crandn((K,1,1),set_mag=1.0) # one dominant path component
                other_paths = 10**(-rice_k_dB/20)*sqrt(.5)*crandn((K,L,1))

                # each of the different paths impinges onto our linear array according to the array spacing and theta
                E = gain*(direct_path + other_paths) * np.exp(1j* theta * np.arange(Nr) )

                # sum the different paths, Z.shape is (user,angle)
                Z = E.sum(axis=1)
                if np.isnan(Z).any():
                    raise RuntimeError()

                # update the data set for these users' signal
                X[i,c,active_users] = np.fft.fft(Z,Nr,axis=-1)/Nr
                ###################################
        # collapse the C and Nu dimensions into one
        X.shape = (batches,C*Nu,Nr)
        if self.ambig:
            X = X[:,np.random.permutation(C*Nu),:]
        if not self.cpx:
            X2 = np.empty( (batches,2*C*Nu,Nr),np.float32)
            X2[:,:C*Nu,:] = X.real
            X2[:,C*Nu:,:] = X.imag
            X = X2
        if self.scramble:
            shp = X.shape
            X = np.random.permutation(X.ravel())
            X.shape = shp
        if self.mmv2d:
            # the "sample vector" dimension should remain in second-to-last dimension
            N = X.shape[-2]
            X = np.reshape( np.transpose(X,(1,0,2)) ,(N,-1) )
        return X

    def fwd(self,X):
        'forward linear operator'
        assert np.iscomplexobj(X) == self.cpx,'wrong value for cpx in constructor'
        return np.einsum('...jk,mj->...mk',X,self.S)

    def adj(self,X):
        'adjoint linear operator'
        assert np.iscomplexobj(Y) == self.cpx,'wrong value for cpx in constructor'
        return np.einsum('...jk,mj->...mk',X,self.S.T.conj())

    def add_noise(self,Y0):
        'add noise at the given SNR, returns Y0+W,wvar'
        wvar = (la.norm(Y0)**2/Y0.size) * 10**(-self.SNR_dB/10)
        if self.cpx:
            Y =(Y0 + crandn(Y0.shape) * sqrt(wvar/2)).astype(np.complex64,copy=False)
        else:
            Y = (Y0 + np.random.normal(scale=sqrt(wvar),size=Y0.shape) ).astype(np.float32,copy=False)
        return Y,wvar

    def genYX(self,batches=1,nsubprocs=None):
        t0 = time.time()
        if nsubprocs is None:
            X = self.genX(batches)
            Y0 = self.fwd(X)
            Y,_ = self.add_noise(Y0)
        else:
            if not hasattr(self,'qgen'):
                import multiprocessing as mp
                self.qgen = mp.Queue(maxsize=nsubprocs) # one slot per subprocess
                def makesets():
                    np.random.seed() #MUST reseed or every subprocess will generate the same data
                    while True:
                        X = self.genX(batches)
                        Y0 = self.fwd(X)
                        Y,_ = self.add_noise(Y0)
                        self.qgen.put((Y,X),block=True)
                self.subprocs = []
                for i in range(nsubprocs):
                    prc = mp.Process(target=makesets)
                    prc.daemon=True
                    prc.start()
                    self.subprocs.append(prc)
            Y,X = self.qgen.get(True)
        et = time.time() - t0
        self.timegen += et
        return (Y,X)

    def kill_subprocs(self):
        if hasattr(self,'qgen') and hasattr(self,'subprocs'):
            for prc in self.subprocs:
                prc.terminate()
                prc.join()
            del self.qgen
            del self.subprocs

if __name__ == '__main__':
    import unittest
    class RapTest(unittest.TestCase):
        def _test_awgn(self,cpx):
            snr = np.random.uniform(3,20)
            p = Problem(cpx=cpx,SNR_dB=snr)
            X = p.genX(5)
            self.assertEqual( np.iscomplexobj(X) , cpx )
            Y0 = p.fwd(X)
            self.assertEqual( np.iscomplexobj(Y0) , cpx )
            Y,wvar = p.add_noise(Y0)
            self.assertEqual( np.iscomplexobj(Y) , cpx )
            snr_obs = -20*np.log10( la.norm(Y-Y0)/la.norm(Y0))
            self.assertTrue( abs(snr-snr_obs) < 1.0, 'gross error in add_noise')
            wvar_obs = la.norm(Y0-Y)**2/Y.size
            self.assertTrue( .5 < wvar_obs/wvar < 1.5, 'gross error in add_noise wvar')

        def test_awgn_cpx(self):
            self._test_awgn(True)

        def test_awgn_real(self):
            self._test_awgn(False)

    unittest.main(verbosity=2)
    #exec(open(os.environ['PYTHONSTARTUP']).read(),globals(),globals())
