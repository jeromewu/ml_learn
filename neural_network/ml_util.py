#!/usr/bin/env python2.7

import numpy as np
import theano
import theano.tensor as T

def softmax(w):
  w = np.array(w)
  maxes = np.amax(w, axis=1)
  maxes = maxes.reshape(maxes.shape[0], 1)
  e = np.exp(w - maxes)
  dist = np.divide(e.T, np.sum(e, axis=1))
  return dist.T

def p_y_given_x(X, w, b):
  dt = np.dot(X, w) + b
  return softmax(dt)

def svm2numpy(file_name, n_dim):
  label = np.array([], dtype=np.int32)
  feature = []
  for line in open(file_name,'r'):
    row = np.zeros(n_dim, dtype=np.float32)
    is_label = True
    for pair in line.rstrip().split(' '):
      if is_label == True:
        label = np.append(label, int(pair))
        is_label = False
        continue
      key = int(pair.split(':')[0])
      val = float(pair.split(':')[1])
      row[key-1] = val
    feature.append(row)
  feature = np.array(feature, dtype=np.float32)
  return (feature, label)

def numpy2svm(x, y, file_name):
  f = open(file_name, 'w')
  for n in xrange(len(x)):
    output_str = str(y[n])
    for idx in xrange(len(x[n])):
      if x[n, idx] != 0.0:
        output_str = ('%s %i:%f') % (output_str, idx+1, x[n, idx])
    f.write(output_str + '\n')

def load_data(train_file_name, test_file_name, n_dim):
  train_set = svm2numpy(train_file_name, n_dim)
  test_set = svm2numpy(test_file_name, n_dim)
  def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')
  train_set_x, train_set_y = shared_dataset(train_set)
  test_set_x, test_set_y = shared_dataset(test_set)
  return train_set_x, train_set_y, test_set_x, test_set_y
