#!/usr/bin/env python2.7

import argparse
import logging
import warnings

import numpy as np
from ml_util import svm2numpy


logger = logging.getLogger( __name__ )

def featdel(input_file_name, n_features, model_file_name):
  logger.info('loading input file: ' + input_file_name)
  x, y = svm2numpy(input_file_name, n_features)
  logger.info('deleting features')
  x_sum = x.sum(axis=0)
  model = []
  for idx in xrange(len(x_sum)):
    if x_sum[idx] == 0.0:
      model.append(idx)
  logger.info(('before deletion: %i, after deletion: %i') % (n_features, n_features - len(model)))
  logger.info('outputing model to file: ' + model_file_name)
  np.save(model_file_name, np.array(model))

def gen_feat(input_file_name, n_features, model_file_name, output_file_name):
  logger.info('loading input file: ' + input_file_name)
  x, y = svm2numpy(input_file_name, n_features)
  logger.info('loading model file: ' + model_file_name)
  model = np.load(model_file_name)
  logger.info('outputing file: ' + output_file_name)
  x = np.delete(x, model, 1)
  f = open(output_file_name, 'w')
  for n in xrange(len(x)):
    output_str = str(y[n])
    for idx in xrange(len(x[n])):
      if x[n, idx] != 0:
        output_str = ('%s %i:%f') % (output_str, idx, x[n, idx])
    f.write(output_str + '\n')

if __name__ == '__main__':
  # logger setup
  logging.basicConfig(level=logging.DEBUG)

  # parser setup
  parser = argparse.ArgumentParser(
    description='Feature Deletion \n' +
                'the data set should be in LIBSVM format(http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)',
    epilog='feature select: \n' +
           __file__ + ' -d -i train.svm -f 784 -m train.npy \n' +
           'generate new feature: \n' +
           __file__ + ' -g -i train.svm -f 784 -m train.npy -o train.new.svm',
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('-d', action='store_true', default=False, help='deletion mode')
  parser.add_argument('-g', action='store_true', default=False, help='generation mode')
  parser.add_argument('-i', action='store', default=None, help='input file name', dest='input_file_name')
  parser.add_argument('-f', action='store', default=0, help='number of features', type=int)
  parser.add_argument('-m', action='store', default=None, help='feature selection model file name', dest='model_file_name')
  parser.add_argument('-o', action='store', default=None, help='output file name', dest='output_file_name')
  args = parser.parse_args()
  if (args.d == False and args.g == False) or (args.d == True and args.g == True):
    logger.error('Unknown mode')
  elif args.d == True:
    logger.info('deletion mode')
    if args.input_file_name == None:
      logger.error('missing input file name')
    elif args.f <= 0:
      logger.error('missing or wrong number of features')
    elif args.model_file_name == None:
      logger.error('missing model file name')
    else:
      featdel(args.input_file_name, args.f, args.model_file_name)
  elif args.g == True:
    logger.info('generation mode')
    if args.input_file_name == None:
      logger.error('missing input file name')
    elif args.f <= 0:
      logger.error('missing or wrong number of features')
    elif args.model_file_name == None:
      logger.error('missing model file name')
    elif args.output_file_name == None:
      logger.error('missing output file name')
    else:
      gen_feat(args.input_file_name, args.f, args.model_file_name, args.output_file_name)
  exit()
