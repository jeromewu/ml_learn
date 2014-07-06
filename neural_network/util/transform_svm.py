#!/usr/bin/env python2
# transform svm features to neural network features

import sys
import numpy as np

def read_svm_from_file(file_name):
  label = np.array([], dtype=np.int32)
  feature_dict_array = []
  n_dim = 0
  for line in open(file_name,'r'):
    feature_dict = {}
    is_label = True
    for pair in line.rstrip().split(' '):
      if is_label == True:
        label = np.append(label, int(pair))
        is_label = False
        continue
      key = int(pair.split(':')[0])
      val = float(pair.split(':')[1])
      feature_dict[key] = val
      if key > n_dim:
        n_dim = key
    feature_dict_array.append(feature_dict)
  return label, feature_dict_array, n_dim

def dict_to_numpy(feature_dict_array, n_dim):
  n_feature = len(feature_dict_array)
  feature = np.zeros((n_feature, n_dim), dtype=np.float32)

  for n in xrange(0, n_feature):
    for idx in feature_dict_array[n].keys():
      feature[n][idx-1] = feature_dict_array[n][idx]

  return feature

def transform_svm(train_file_name, test_file_name):
  train_label, train_feature_dict_array, train_n_dim = read_svm_from_file(train_file_name)
  test_label, test_feature_dict_array, test_n_dim = read_svm_from_file(test_file_name)

  n_dim = max([train_n_dim, test_n_dim])
  n_label = max(max(train_label), max(test_label)) + 1

  train_feature = dict_to_numpy(train_feature_dict_array, n_dim)
  test_feature = dict_to_numpy(test_feature_dict_array, n_dim)

  #slim svm feature
  del_arr = []
  train_feature_sum = train_feature.sum(axis=0)
  test_feature_sum = test_feature.sum(axis=0)
  total_feature_sum = train_feature_sum + test_feature_sum
  for i in xrange(0,len(total_feature_sum)):
    if total_feature_sum[i] == 0.0:
      del_arr.append(i)

  train_feature = np.delete(train_feature, del_arr, 1)
  test_feature = np.delete(test_feature, del_arr, 1)
  n_dim -= len(del_arr)
  return (train_feature, train_label), (test_feature, test_label), n_dim, n_label

if __name__=='__main__':
  train_data, test_data, n_dim, n_label = transform_svm(sys.argv[1], sys.argv[2])
  print n_dim, n_label
