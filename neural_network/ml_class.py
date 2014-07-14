#!/usr/bin/env python2.7

import logging

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

logger = logging.getLogger( __name__ )

class LogisticRegression(object):
  def __init__(self, input, n_in, n_out, W=None, b=None):
    logger.debug(('sgd:n_in=%i, n_out=%i') % (n_in, n_out))
    if W == None or b == None:
      W = np.zeros((n_in, n_out), dtype=theano.config.floatX)
      b = np.zeros((n_out,), dtype=theano.config.floatX)
    self.input = input
    self.W = theano.shared(value=W, name='W', borrow=True)
    self.b = theano.shared(value=b, name='b', borrow=True)
    self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
    self.y_pred = T.argmax(self.p_y_given_x, axis=1)
    self.params = [self.W, self.b]

  def negative_log_likelihood(self, y):
    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

  def errors(self, y):
    if y.ndim != self.y_pred.ndim:
      raise TypeError('y should have the same shape as self.y_pred',
                      ('y', target.type, 'y_pred', self.y_pred.type))
    if y.dtype.startswith('int'):
      return T.mean(T.neq(self.y_pred, y))
    else:
      raise NotImplementedError()

class HiddenLayer(object):
  def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
    logger.debug(('hidden layer:n_in=%i, n_out=%i') % (n_in, n_out))
    self.input = input
    if W is None:
      W = np.asarray(rng.uniform(
                     low=-np.sqrt(6. / (n_in + n_out)),
                     high=np.sqrt(6. / (n_in + n_out)),
                     size=(n_in, n_out)), dtype=theano.config.floatX)
      if activation == T.nnet.sigmoid:
        W *= 4
    if b is None:
      b = np.zeros((n_out,), dtype=theano.config.floatX)
    self.W = theano.shared(value=W, name='W', borrow=True)
    self.b = theano.shared(value=b, name='W', borrow=True)
    lin_output = T.dot(input, self.W) + self.b
    self.output = (lin_output if activation is None else activation(lin_output))
    self.params = [self.W, self.b]

class MLP(object):
  def __init__(self, rng, input, n_in, n_hidden, n_out, hidden_W=None, hidden_b=None, sgd_W=None, sgd_b=None):
    self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, W=hidden_W, b=hidden_b, activation=T.tanh)
    self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out, W=sgd_W, b=sgd_b)
    self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
    self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()
    self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
    self.errors = self.logRegressionLayer.errors
    self.params = self.hiddenLayer.params + self.logRegressionLayer.params

class LeNetConvPoolLayer(object):
  def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
    assert image_shape[1] == filter_shape[1]
    self.input = input
    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)
    b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
    self.b = theano.shared(value=b_values, borrow=True)
    conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
    pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_boarder=True)
    self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    self.params = [self.W, self.b]
