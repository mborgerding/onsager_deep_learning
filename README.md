# What is this?

This project contains scripts to reproduce experiments from the paper
[Onsager-Corrected Deep Networks for Sparse Linear Inverse Problems](https://arxiv.org/pdf/1612.01183)
by 
[Mark Borgerding](mailto://borgerding.7@osu.edu)
and 
[Phil](mailto://schniter.1@osu.edu)
[Schniter](http://www2.ece.ohio-state.edu/~schniter)


# The Problem of Interest

Briefly, the _Sparse Linear Inverse Problem_ is the estimation of an unknown signal from indirect, noisy, underdetermined measurements by exploiting the knowledge that the signal has many zeros.  We compare various iterative algorithmic approaches to this problem and explore how they benefit from loop-unrolling and deep learning.

# Overview

The included scripts 
- are generally written in python and require [TensorFlow](http://www.tensorflow.org),
- work best with a GPU,
- generate synthetic data as needed,
- are known to work with CentOS 7 Linux and TensorfFlow 0.9,
- may sometimes be written in octave/matlab .m files.

# Description of Files

## [save_problem.py](save_problem.py) 

Creates numpy archives (.npz) and matlab (.mat) files with (y,x,A) for the sparse linear problem y=Ax+w.
These files are not really necessary for any of the deep-learning scripts, which generate the problem on demand.
They are merely provided for better understanding the specific realizations used in the experiments.

e.g.
```
$ ./save_problem.py 
...
saved numpy archive problem_Giid.npz
saved matlab file problem_Giid.mat
...
saved numpy archive problem_k15.npz
saved matlab file problem_k15.mat
```

## [ista_fista_amp.m](ista_fista_amp.m)

Using the .mat files created by save_problem.py, this octave/matlab script tests the performance of non-learned algorithms ISTA, FISTA, and AMP.

e.g.
```
octave:1> ista_fista_amp
loaded Gaussian A problem
AMP reached NMSE=-35dB at iteration 25
AMP terminal NMSE=-36.7304 dB
FISTA reached NMSE=-35dB at iteration 202
FISTA terminal NMSE=-36.7415 dB
ISTA reached NMSE=-35dB at iteration 3420
ISTA terminal NMSE=-36.7419 dB
```

## [lista.py](lista.py)

This is an implementation of LISTA _Learned Iterative Soft Thresholding Algorithm_ by (Gregor&LeCun, 2010 ICML).

e.g. To reproduce the `LISTA` trace from Fig.9,
```
$ ./lista.py --T 1 --save /tmp/T1.npz --trainRate=1e-3 --refinements=4 --stopAfter=20 
...
step=15990 elapsed=133.239667892 nic=319 nmse_test=-6.40807334129 nmse_val=-6.46110795422 nmse_val0=-0.806287242 
no improvement in 320 checks,Test NMSE=-6.408dB
$ for T in {2..20};do ./lista.py --T $T --save /tmp/T${T}.npz --load /tmp/T$(($T-1)).npz --setup --trainRate=1e-3 --refinements=4 --stopAfter=20 --summary /tmp/${T}.sum || break ;done
...
```
The `nmse_val` is the quantity that is plotted in the paper. It is from a mini-batch that is used for training or iany decisions. The `nmse_test` is from a minibatch that is also not trained, but it _is_ used for decisions about training step size, and termination criteria.  This convention holds for all experiments.

## [lamp_vamp.py](lamp_vamp.py)

Learns the parameters for Learned AMP (LAMP) or Vector AMP(VAMP) with a variety of shrinkage functions.
This script may be called independently or from run_lamp_vamp.sh.

e.g. The following generates the `matched VAMP` trace from Fig.12
```
$ for T in {1..15};do ./lamp_vamp.py --matched --T $T --summary=matched${T}.sum;done
...
$ for T in {1..15};do echo -n "T=$T "; grep nmse_val= matched${T}.sum;done
T=1 nmse_val=-6.74708897423
T=2 nmse_val=-12.5694582254
T=3 nmse_val=-18.8778007058
T=4 nmse_val=-25.7153599678
T=5 nmse_val=-32.8098204058
T=6 nmse_val=-39.1792426565
T=7 nmse_val=-43.3195721343
T=8 nmse_val=-44.9222227945
T=9 nmse_val=-45.3680144768
T=10 nmse_val=-45.4783550406
T=11 nmse_val=-45.4985886728
T=12 nmse_val=-45.5054164287
T=13 nmse_val=-45.5063294603
T=14 nmse_val=-45.50776381
T=15 nmse_val=-45.5077351689
```
Here, the `--matched` argument bypasses training by forcing some argument values, specifically `--vamp --shrink bg --trainRate=0`.

## [run_lamp_vamp.sh](run_lamp_vamp.sh)

bash script to drive lamp_vamp.py with different shrinkage functions, algorithms, matrix types, etc.
This takes days to run, even with a fast GPU.

## [let_vamp_off_leash.py](let_vamp_off_leash.py)

This demonstrates that matched VAMP represents a fixed point for a Learned VAMP (LVAMP) network.
The network is given the benefit of a large number of parameters.
One might expect that deep learning/backpropagation would yield some improvements over the prescribed structure and values of 
VAMP.

Notably we find that **backpropagation yields no improvement to LVAMP when initialized with matched parameters**.
```
$ ./let_vamp_off_leash.py --T 6  --trainRate=1e-5  --refinements=0 --stopAfter=200
...
step=10 elapsed=0.605620861053 nic=1 nmse_test=-38.9513696476 nmse_val=-39.2820689456 nmse_val0=-39.2881639853
...
step=1990 elapsed=57.8836979866 nic=199 nmse_test=-38.9146799477 nmse_val=-39.2407649471 nmse_val0=-39.2881639853 
```

Then the initialization is slightly perturbed away from matched parameters such that the initial performrance is almost 6dB
worse.
We see that backpropagation does its job and finds its way back to
approximately the same level as with the matched parameters.  The slight difference is explainable by different realizations of the training minibatches.
```
$ ./let_vamp_off_leash.py --T 6  --vamp --trainRate=1e-5  --refinements=0 --stopAfter=200 --errTheta .2 --errV 1e-3
...
initial test_nmse=-33.612dB val_nmse=-33.691dB
step=10 elapsed=0.604254961014 nic=0 nmse_test=-35.3966007584 nmse_val=-35.5191812551 nmse_val0=-33.6907302155
...
step=20100 elapsed=586.414004087 nic=199 nmse_test=-39.0699940418 nmse_val=-39.369836192 nmse_val0=-33.6907302155 
```


## [shrinkage.py](shrinkage.py)
	python module which defines the shrinkage functions we investigated and parameterized
	- soft-threshold (scaled)
	- piecewise-linear 
	- exponential
	- spline-based
	- Bernoulli-Gaussian MMSE

## [utils.py](utils.py)

	Various python/tensorflow utilities.

## [utils.sh](utils.sh)

	Various shell script utility functions.

