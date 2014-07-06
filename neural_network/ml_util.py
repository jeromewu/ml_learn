#!/usr/bin/env python2.7

import numpy as np

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
