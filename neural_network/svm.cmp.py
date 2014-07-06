#!/usr/bin/python2

from lib.svmutil import *
import sys

y,x = svm_read_problem(sys.argv[1])

FP = 0 #0->1
FN = 0 #1->0
cnt = 0
n_malware = 0
n_benign = 0

for line in open(sys.argv[2],'r'):
  if int(y[cnt]) == 0 and int(line) == 1:
    FP+=1
  elif int(y[cnt]) == 1 and int(line) == 0:
    FN+=1
  if int(line) == 0:
    n_benign+=1
  if int(line) == 1:
    n_malware+=1
  cnt+=1
fp = float(FP)/float(n_benign)
fn = float(FN)/float(n_malware)
recall = 1-fn
precision = 1-fp
fscore = (2*recall*precision)/(recall+precision)
print fscore,recall,precision,fp,fn
