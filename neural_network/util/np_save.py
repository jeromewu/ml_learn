#!/usr/bin/env python2.7

import numpy as np
import theano

W = theano.shared(value=np.zeros((10, 20), dtype=theano.config.floatX), 
                  name='W',
                  borrow=True)
b = theano.shared(value=np.zeros((1, 2), dtype=theano.config.floatX), 
                  name='b',
                  borrow=True)

np.save('test.npy',(W.get_value(), b.get_value()))
iW, ib = np.load('test.npy')
print iW, ib
