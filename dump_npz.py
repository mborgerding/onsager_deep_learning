#!/usr/bin/python
from __future__ import division,print_function

import numpy as np
import sys

if len(sys.argv)<=1:
    sys.stderr.write('provide at least one .npz file to show\n')
    sys.exit(1)

for f in sys.argv[1:]:
    print('Loading '+f)
    for k,v in sorted(np.load(f).items()):
        print('{}:{}'.format(k,v))

