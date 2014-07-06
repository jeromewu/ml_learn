#!/usr/bin/env python2.7

import argparse
import logging
import os.path
import sys
import time

import numpy as np
import theano
import theano.tensor as T
import ml_util

logger = logging.getLogger( __name__ )

class LogisticRegression(object):
  def __init__(self, input, n_in, n_out, model_file_name):
    if os.path.isfile(model_file_name):
      W_val, b_val = np.load(model_file_name)
    else:
      W_val = np.zeros((n_in, n_out), dtype=theano.config.floatX)
      b_val = np.zeros((n_out,), dtype=theano.config.floatX)
    self.input = input
    self.W = theano.shared(value=W_val, name='W', borrow=True)
    self.b = theano.shared(value=b_val, name='b', borrow=True)
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

def get_y_pred(model_file_name, test_set_x):
  W, b = np.load(model_file_name)
  _p_y_given_x = ml_util.p_y_given_x(test_set_x, W, b)
  return np.argmax(_p_y_given_x, axis=1)

def load_data(train_file_name, test_file_name, n_dim):
  train_set = ml_util.svm2numpy(train_file_name, n_dim)
  test_set = ml_util.svm2numpy(test_file_name, n_dim)
  def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')
  train_set_x, train_set_y = shared_dataset(train_set)
  test_set_x, test_set_y = shared_dataset(test_set)
  return train_set_x, train_set_y, test_set_x, test_set_y


def sgd_train(train_file_name, test_file_name, model_file_name, n_dim, n_label, learning_rate=0.13, batch_size=600):
  logger.info('loading data')
  train_set_x, train_set_y, test_set_x, test_set_y = load_data(train_file_name, test_file_name, n_dim)
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

  logger.info('building the model')
  idx = T.lscalar()
  x = T.matrix('x')
  y = T.ivector('y')

  classifier = LogisticRegression(input=x, n_in=n_dim, n_out=n_label, model_file_name=model_file_name)
  cost = classifier.negative_log_likelihood(y)

  test_model = theano.function(inputs=[idx],
                               outputs=classifier.errors(y),
                               givens={
                                 x: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                                 y: test_set_y[idx * batch_size: (idx + 1) * batch_size]
                               })
  g_W = T.grad(cost=cost, wrt=classifier.W)
  g_b = T.grad(cost=cost, wrt=classifier.b)
  updates = [(classifier.W, classifier.W - learning_rate * g_W),
             (classifier.b, classifier.b - learning_rate * g_b)]
  train_model = theano.function(inputs=[idx],
                                outputs=cost,
                                updates=updates,
                                givens={
                                  x: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                                  y: train_set_y[idx * batch_size: (idx + 1) * batch_size]
                                })
  logger.info('training the model')
  best_test_score = np.inf
  epoch = 0
  start_time = time.clock()
  while True:
    epoch = epoch + 1
    for minibatch_idx in xrange(n_train_batches):
      train_model(minibatch_idx)
      iter = (epoch - 1) * n_train_batches + minibatch_idx
      if (iter + 1) % n_train_batches == 0:
        test_loss = [test_model(i) for i in xrange(n_test_batches)]
        test_score = np.mean(test_loss)
        logger.info(('epoch %i, ran for %.5f hr, minibatch %i/%i, test error %f %%') % (epoch, (time.clock() - start_time)/3600.0, minibatch_idx + 1, n_train_batches, test_score * 100.))
        if test_score < best_test_score:
          np.save(model_file_name, (classifier.W.get_value(), classifier.b.get_value()))
          best_test_score = test_score
          logger.info(('epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_idx + 1, n_train_batches, best_test_score * 100.))

def sgd_predict(model_file_name, test_file_name, pred_file_name, n_dim):
  logger.info('loading test file')
  (test_set_x, test_set_y) = ml_util.svm2numpy(test_file_name, n_dim)
  logger.info('predicting')
  f = open(pred_file_name, 'w')
  for label in get_y_pred(model_file_name, test_set_x):
    f.write(str(label)+'\n')

if __name__ == '__main__':
  # logger setup
  logging.basicConfig(level=logging.INFO)

  # parser setup
  parser = argparse.ArgumentParser(
    description='Logistic SGD GPU version \n' +
                'the data set should be in LIBSVM format(http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)',
    epilog='train model: \n' +
           __file__ + ' -t --train train.svm --test test.svm --model model.npy -d 784 -l 10 \n' +
           'prediction: \n' +
           __file__ + ' -p --model model.npy --test test.svm --pred pred.out -d 784 -l 10',
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('-t', action='store_true', default=False, help='train mode')
  parser.add_argument('-p', action='store_true', default=False, help='prediction mode')
  parser.add_argument('--train', action='store', default=None, help='train file name', dest='train_file_name')
  parser.add_argument('--test', action='store', default=None, help='test file name', dest='test_file_name')
  parser.add_argument('--model', action='store', default=None, help='model file name', dest='model_file_name')
  parser.add_argument('--pred', action='store', default=None, help='predict file name', dest='pred_file_name')
  parser.add_argument('-d', action='store', default=0, help='length of feature dimension', type=int)
  parser.add_argument('-l', action='store', default=0, help='number of label', type=int)
  args = parser.parse_args()

  if (args.t == False and args.p == False) or (args.t == True and args.p == True):
    logger.info('Unknown mode')
    exit()

  if args.t == True:
    logger.info('train mode')
    if args.train_file_name == None:
      logger.info('missing train file name')
    elif args.test_file_name == None:
      logger.info('missing test file name')
    elif args.model_file_name == None:
      logger.info('missing model file name')
    elif args.d <= 0:
      logger.info('missing or wrong length of feature dimension')
    elif args.l <= 0:
      logger.info('missing or wrong number of label')
    else:
      sgd_train(args.train_file_name, args.test_file_name, args.model_file_name, args.d, args.l)
  elif args.p == True:
    logger.info('prediction mode')
    if args.model_file_name == None:
      logger.info('missing model file name')
    elif args.test_file_name == None:
      logger.info('missing test file name')
    elif args.pred_file_name == None:
      logger.info('missing pred file name')
    elif args.d <= 0:
      logger.info('missing or wrong length of feature dimension')
    elif args.l <= 0:
      logger.info('missing or wrong number of label')
    else:
      sgd_predict(args.model_file_name, args.test_file_name, args.pred_file_name, args.d)
  exit()
