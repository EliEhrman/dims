"""
This module evolved from knn which evolved from dims
The goal of this modified version is to estimate how much time the gpgpu takes to do a kmin

"""

import numpy as np
import tensorflow as tf
import csv
import random
import math
import time
import os

num_centroids = 2
num_cluster_iters = 2
num_clusters_to_get = 2
num_ks = 200
num_gpupu_ops = 1000

knn_in_fname = '/devlink2/data/imagenet/imgnet_small.txt'
csvfile = open(knn_in_fname, 'rt')
reader = csv.reader(csvfile, delimiter=',')
b_fname_included = True


allrows = []
labels = []

data_start_col = 1
if b_fname_included:
	data_start_col = 2

for row in reader:
	labels.append(row[0])
	vals = [float(d) for d in row[data_start_col:]]
	allrows.append(vals)

csvfile.close()

labels_unique = set(labels)
num_labels = len(labels_unique)

allcols = zip(*allrows)
mins = [min(col) for col in allcols]
maxs = [max(col) for col in allcols]
allvals = []
for row in allrows:
	vals = []
	for icol, col in enumerate(row):
		vals.append((col - mins[icol]) / (maxs[icol] - mins[icol]))
	sqrtsum = math.sqrt(sum([x ** 2 for x in vals]))
	allvals.append([x / sqrtsum for x in vals])

reclen = len(allvals[0])
numrecs = len(allvals)

random.seed(0)
shuffle_stick = [i for i in range(numrecs)]
random.shuffle(shuffle_stick)
allvals = [allvals[i] for i in shuffle_stick]
labels = [labels[i] for i in shuffle_stick]

lasttrain = 4 * numrecs / 5
trainvals = allvals[:lasttrain]
testvals = allvals[lasttrain:]


batch_size = lasttrain

trainx = tf.placeholder(tf.float32, [batch_size, reclen])
testx = tf.placeholder(tf.float32, [reclen])
trainl = tf.placeholder(tf.int32, [batch_size])

CDs1 = tf.reduce_sum(tf.mul(trainx, testx), axis=1)
def add_ops():
	bestCDs, bestCDIDxs0 = tf.nn.top_k(tf.mul(CDs1, 7.0), num_ks)
	for i in range(num_gpupu_ops):
		bestCDs1, bestCDIDxs1 = tf.nn.top_k(tf.mul(CDs1, float(i)), num_ks)
		bestCDIDxs0 = tf.add(bestCDIDxs0, bestCDIDxs1)
	return bestCDIDxs0

bestCDIDxs = add_ops()



sess = tf.Session()
sess.run(tf.global_variables_initializer())


num_errors = 0
num_steps = 1
timespent = 0.0
dummy_vals = []
for step in range(num_steps):
	for itestitem, testitem in enumerate(testvals):


		fd = {trainx: trainvals, testx: testitem}
		starttime = time.time()
		bestCDIDxs_vals, = sess.run([bestCDIDxs], feed_dict=fd)
		endtime = time.time()
		timespent += (endtime - starttime)

		dummy_vals.append( bestCDIDxs_vals)

	print len(dummy_vals)
	print 'Each call took ', timespent / len(testvals), ' s'

sess.close()






print 'done'