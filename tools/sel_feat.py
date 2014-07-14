#!/usr/bin/env python2.7

import argparse
import logging
import warnings

from ml_util import svm2numpy
from PyML import *
import numpy as np

logger = logging.getLogger( __name__ )

def sel_feat(input_file_name, model_file_name):
  logger.info('loading input file: ' + input_file_name)
  data = SparseDataSet(input_file_name)
  logger.info('selecting features')
  rfe = featsel.RFE()
  rfe.select(data)
  logger.info(('after selection: %i') % (len(data.featureID)))
  logger.info('outputing model to file: ' + model_file_name)
  np.save(model_file_name, np.array(data.featureID))

def gen_feat(input_file_name, n_features, model_file_name, output_file_name):
  logger.info('loading input file: ' + input_file_name)
  x, y = svm2numpy(input_file_name, n_features)
  logger.info('loading model file: ' + model_file_name)
  mask_idx = np.load(model_file_name)
  logger.info('len after selection: %i' % (len(mask_idx)))
  logger.info('outputing file: ' + output_file_name)
  f = open(output_file_name, 'w')
  for n in xrange(x.shape[0]):
    output_str = str(y[n])
    ptr = 1
    for idx in mask_idx:
      if x[n, int(idx)-1] != 0.0:
        output_str = ('%s %i:%f') % (output_str, ptr, x[n, int(idx)-1])
      ptr += 1
    f.write(output_str + '\n')

if __name__ == '__main__':
  # logger setup
  logging.basicConfig(level=logging.DEBUG)

  # parser setup
  parser = argparse.ArgumentParser(
    description='Feature Selection base on scikit-learn package \n' +
                'the data set should be in LIBSVM format(http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)',
    epilog='feature select: \n' +
           __file__ + ' -s -i train.svm -m train.npy \n' +
           'generate new feature: \n' +
           __file__ + ' -g -i train.svm -f 784 -m train.npy -o train.new.svm',
    formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('-s', action='store_true', default=False, help='selection mode')
  parser.add_argument('-g', action='store_true', default=False, help='generation mode')
  parser.add_argument('-i', action='store', default=None, help='input file name', dest='input_file_name')
  parser.add_argument('-f', action='store', default=0, help='number of features', type=int)
  parser.add_argument('-m', action='store', default=None, help='feature selection model file name', dest='model_file_name')
  parser.add_argument('-o', action='store', default=None, help='output file name', dest='output_file_name')
  args = parser.parse_args()
  if (args.s == False and args.g == False) or (args.s == True and args.g == True):
    logger.error('Unknown mode')
  elif args.s == True:
    logger.info('selection mode')
    if args.input_file_name == None:
      logger.error('missing input file name')
    elif args.model_file_name == None:
      logger.error('missing model file name')
    else:
      sel_feat(args.input_file_name, args.model_file_name)
  elif args.g == True:
    logger.info('generation mode')
    if args.input_file_name == None:
      logger.error('missing input file name')
    elif args.model_file_name == None:
      logger.error('missing model file name')
    elif args.output_file_name == None:
      logger.error('missing output file name')
    elif args.f <= 0:
      logger.error('missing or wrong number of features')
    else:
      gen_feat(args.input_file_name, args.f, args.model_file_name, args.output_file_name)
  exit()
