#!/usr/bin/env python2.7

import argparse
import logging
import os.path
import time

import numpy as np
import theano
import theano.tensor as T
from ml_util import load_data, svm2numpy, p_y_given_x
from ml_class import MLP

logger = logging.getLogger( __name__ )

def get_y_pred(model_file_name, test_set_x):
  hidden_W, hidden_b, sgd_W, sgd_b = np.load(model_file_name)
  hidden_output = np.tanh(np.dot(test_set_x, hidden_W) + hidden_b)
  _p_y_given_x = p_y_given_x(hidden_output, sgd_W, sgd_b)
  return np.argmax(_p_y_given_x, axis=1)

def mlp_train(train_file_name, test_file_name, model_file_name, n_dim, n_label, n_hidden, learning_rate, L1_reg=0.00, L2_reg=0.0001, batch_size=20):
  logger.info('loading data')
  train_set_x, train_set_y, test_set_x, test_set_y = load_data(train_file_name, test_file_name, n_dim)
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
  logger.info('building the model')
  idx = T.lscalar()
  x = T.matrix('x')
  y = T.ivector('y')
  rng = np.random.RandomState(1234)
  
  hidden_W = None
  hidden_b = None
  sgd_W = None
  sgd_b = None
  if os.path.isfile(model_file_name):
    logger.debug('load existing model')
    hidden_W, hidden_b, sgd_W, sgd_b = np.load(model_file_name)
  
  classifier = MLP(rng=rng, input=x, n_in=n_dim, n_hidden=n_hidden, n_out=n_label, hidden_W=hidden_W, hidden_b=hidden_b, sgd_W=sgd_W, sgd_b=sgd_b)
  cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
  test_model = theano.function(inputs=[idx], 
                               outputs=classifier.errors(y), 
                               givens={
                                 x: test_set_x[idx * batch_size:(idx + 1) * batch_size],
                                 y: test_set_y[idx * batch_size:(idx + 1) * batch_size]
                              })
  gparams = []
  for param in classifier.params:
    gparam = T.grad(cost, param)
    gparams.append(gparam)
  updates = []
  for param, gparam in zip(classifier.params, gparams):
    updates.append((param, param - learning_rate * gparam))
  train_model = theano.function(inputs=[idx], 
                                outputs=cost,
                                updates=updates,
                                givens={
                                  x: train_set_x[idx * batch_size:(idx + 1) * batch_size],
                                  y: train_set_y[idx * batch_size:(idx + 1) * batch_size]
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
          np.save(model_file_name, (classifier.hiddenLayer.W.get_value(), classifier.hiddenLayer.b.get_value(), classifier.logRegressionLayer.W.get_value(), classifier.logRegressionLayer.b.get_value()))
          best_test_score = test_score
          logger.info(('epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_idx + 1, n_train_batches, best_test_score * 100.))


def mlp_predict(model_file_name, test_file_name, pred_file_name, n_dim):
  logger.info('loading test file')
  (test_set_x, test_set_y) = svm2numpy(test_file_name, n_dim)
  logger.info('predicting')
  f = open(pred_file_name, 'w')
  for label in get_y_pred(model_file_name, test_set_x):
    f.write(str(label)+'\n')


if __name__ == '__main__':
  # logger setup
  logging.basicConfig(level=logging.DEBUG)

  # parser setup
  parser = argparse.ArgumentParser(
    description='MLP GPU version \n' +
                'the data set should be in LIBSVM format(http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)',
    epilog='train model: \n' +
           __file__ + ' -t --train train.svm --test test.svm --model model.npy -f 784 -l 10 --hunits 784 -r 0.01\n' +
           'prediction: \n' +
           __file__ + ' -p --model model.npy --test test.svm --pred pred.out -f 784 -l 10',
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('-t', action='store_true', default=False, help='train mode')
  parser.add_argument('-p', action='store_true', default=False, help='prediction mode')
  parser.add_argument('--train', action='store', default=None, help='train file name', dest='train_file_name')
  parser.add_argument('--test', action='store', default=None, help='test file name', dest='test_file_name')
  parser.add_argument('--model', action='store', default=None, help='model file name', dest='model_file_name')
  parser.add_argument('--pred', action='store', default=None, help='predict file name', dest='pred_file_name')
  parser.add_argument('-f', action='store', default=0, help='number of features', type=int)
  parser.add_argument('-l', action='store', default=0, help='number of label', type=int)
  parser.add_argument('--hunits', action='store', default=0, help='number of hidden units', type=int)
  parser.add_argument('-r', action='store', default=0.1, help='learning rate, default=0.1', type=float)
  args = parser.parse_args()

  if (args.t == False and args.p == False) or (args.t == True and args.p == True):
    logger.error('Unknown mode')
  elif args.t == True:
    logger.info('train mode')
    if args.train_file_name == None:
      logger.error('missing train file name')
    elif args.test_file_name == None:
      logger.error('missing test file name')
    elif args.model_file_name == None:
      logger.error('missing model file name')
    elif args.f <= 0:
      logger.error('missing or wrong length of feature dimension')
    elif args.l <= 0:
      logger.error('missing or wrong number of label')
    elif args.hunits <= 0:
      logger.error('missing or wrong number of hidden units')
    elif args.r <= 0:
      logger.error('wrong learning rate')
    else:
      mlp_train(args.train_file_name, args.test_file_name, args.model_file_name, args.f, args.l, args.hunits, args.r)
  elif args.p == True:
    logger.info('prediction mode')
    if args.model_file_name == None:
      logger.error('missing model file name')
    elif args.test_file_name == None:
      logger.error('missing test file name')
    elif args.pred_file_name == None:
      logger.error('missing pred file name')
    elif args.f <= 0:
      logger.error('missing or wrong length of feature dimension')
    elif args.l <= 0:
      logger.error('missing or wrong number of label')
    else:
      mlp_predict(args.model_file_name, args.test_file_name, args.pred_file_name, args.f)
  exit()
