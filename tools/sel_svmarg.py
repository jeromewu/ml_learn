#!/usr/bin/env python2.7

import argparse
import logging
import multiprocessing
import math

from lib.svmutil import *

logger = logging.getLogger( __name__ )

fw_lock = multiprocessing.Lock()
cnt_lock = multiprocessing.Lock()
cnt = multiprocessing.Value('i', 0)


def inc_cnt():
  with cnt_lock:
    cnt.value += 1
    return cnt.value

def file_write(log_file_name, msg):
  if log_file_name != None:
    with fw_lock:
      file_append = open(log_file_name,'a')
      file_append.write(msg)
      file_append.close()

def worker_in(train_y, train_x, args, log_file_name):
  args_str = ('-q -v %i -c %f -g %f') % (args['n_folds'], args['c'], args['g'])
  acc = svm_train(train_y, train_x, args_str)
  logger.info(('jobs finished: %i/%i, acc=%f, c=%f, g=%f') % (inc_cnt(), args['n_jobs'], acc, args['c'], args['g']))
  file_write(log_file_name, ('%f %f %f') % (acc, args['c'], args['g']))
  return {'acc':acc, 'c':args['c'], 'g':args['g']}

def worker_out(train_y, train_x, test_y, test_x, args, log_file_name):
  args_str = ('-q -c %f -g %f') % (args['c'], args['g'])
  model = svm_train(train_y, train_x, args_str)
  label, (acc, mse, scc), val = svm_predict(test_y, test_x, model)
  logger.info(('jobs finished: %i/%i, acc=%f, c=%f, g=%f') % (inc_cnt(), args['n_jobs'], acc, args['c'], args['g']))
  file_write(log_file_name, ('%f %f %f') % (acc, args['c'], args['g']))
  return {'acc':acc, 'c':args['c'], 'g':args['g']}

def sel_svmarg_in(train_file_name=None, log_file_name=None, n_folds=0, c_start=-5, c_end=16, c_step=2, g_start=3, g_end=-16, g_step=-2):
  pool = multiprocessing.Pool()
  logger.info('loading files')
  train_y, train_x = svm_read_problem(train_file_name)
  logger.info('start selecting arguments')
  out_queue = multiprocessing.Queue()
  jobs = []
  c_list = xrange(c_start, c_end, c_step)
  g_list = xrange(g_start, g_end, g_step)
  for c in c_list:
    for g in g_list:
      args = {'n_folds':n_folds, 'c':pow(2, c), 'g':pow(2, g), 'n_jobs':len(c_list)*len(g_list)}
      p = pool.apply_async(worker_in, (train_y, train_x, args, log_file_name, ))
      jobs.append(p)
 
  ret = []
  for i in xrange(len(jobs)):
    ret.append(jobs[i].get())

  max_ret = max(ret, key=lambda k:k['acc'])
  logger.info(('max acc=%f, c=%f, g=%f') % (max_ret['acc'], max_ret['c'], max_ret['g']))

def sel_svmarg_out(train_file_name=None, test_file_name=None, log_file_name=None, c_start=-5, c_end=16, c_step=2, g_start=3, g_end=-16, g_step=-2):
  pool = multiprocessing.Pool()
  logger.info('loading files')
  train_y, train_x = svm_read_problem(train_file_name)
  test_y, test_x = svm_read_problem(test_file_name)
  logger.info('start selecting arguments')
  jobs = []
  c_list = xrange(c_start, c_end, c_step)
  g_list = xrange(g_start, g_end, g_step)
  for c in c_list:
    for g in g_list:
      args = {'c':pow(2, c), 'g':pow(2, g), 'n_jobs':len(c_list)*len(g_list)}
      p = pool.apply_async(worker_out, (train_y, train_x, test_y, test_x, args, log_file_name, ))
      jobs.append(p)
  
  ret = []
  for i in xrange(len(jobs)):
    ret.append(jobs[i].get())
  
  max_ret = max(ret, key=lambda k:k['acc'])
  logger.info(('max acc=%f, c=%f, g=%f') % (max_ret['acc'], max_ret['c'], max_ret['g']))

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  parser = argparse.ArgumentParser(
    description='SVM argument selection parallel version, focus on c and g value only \n' +
                'the data set should be in LIBSVM format(http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)',
    epilog='SVM in-sample argument selection: \n' +
           __file__ + ' --train train.dat --log arg.log -v 5 \n' +
           'SVM out-sample argument selection: \n' +
           __file__ + ' --train train.dat --test test.dat',
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--train', action='store', default=None, help='train file name', dest='train_file_name')
  parser.add_argument('--test', action='store', default=None, help='test file name', dest='test_file_name')
  parser.add_argument('--log', action='store', default=None, help='log file name', dest='log_file_name')
  parser.add_argument('-v', action='store', default=0, help='number of folds', dest='n_folds', type=int)
  args = parser.parse_args()

  if args.n_folds > 0:
    logger.info('In-sample mode')
    if args.train_file_name == None:
      logger.error('Missing train file name')
    else:
      sel_svmarg_in(train_file_name=args.train_file_name, n_folds=args.n_folds, log_file_name=args.log_file_name)
  elif args.n_folds == 0:
    logger.info('Out-sample mode')
    if args.train_file_name == None:
      logger.error('Missing train file name')
    elif args.test_file_name == None:
      logger.error('Missing test file name')
    else:
      sel_svmarg_out(train_file_name=args.train_file_name, test_file_name=args.test_file_name, log_file_name=args.log_file_name)
  else:
    logger.error('Wrong number of folds')
