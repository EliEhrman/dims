"""
Basic implementation of KNN in tensorflow for the purpose of speedingup python and taking advantage of the gpgpu

Clusters code present but only activated if b_use_clusters is True
However, for now, all candidates and not just the clusters are passed in

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


knn_in_fname = '/devlink2/data/imagenet/imgnet_small.txt'
csvfile = open(knn_in_fname, 'rt')
reader = csv.reader(csvfile, delimiter=',')
b_use_clusters = False
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

if b_use_clusters:
	centroids = []
	for icent in range(num_centroids):
		centroids.append(trainvals[random.randint(0, len(trainvals))])

	for iiter in range(num_cluster_iters):
		new_centroids = [[0.0 for val in centroid] for centroid in centroids]
		num_in_cluster = [0 for i in range(num_centroids)]

		for dbitem in trainvals:
			CDs = [sum([val * dbitem[ival] for ival, val in enumerate(centroid)]) for centroid in centroids ]
			i_best_centroid = np.argmax(np.asarray(CDs))
			new_centroids[i_best_centroid] = [new_centroids[i_best_centroid][ival] + val for ival, val in enumerate(dbitem)]
			num_in_cluster[i_best_centroid] += 1

		centroids = [[val / num_in_cluster[icentroid] if num_in_cluster[icentroid] > 0  else 0.0 for val in centroid] for icentroid, centroid in enumerate(new_centroids)]

	clusters = [[] for i in range(num_centroids)]
	# labels_for_clusters = [[] for i in range(num_centroids)]

	for iitem, dbitem in enumerate(trainvals):
		CDs = [sum([val * dbitem[ival] for ival, val in enumerate(centroid)]) for centroid in centroids]
		i_best_centroid = np.argmax(np.asarray(CDs))
		clusters[i_best_centroid].append(iitem)
		# labels_for_clusters[i_best_centroid].append(labels[iitem])

batch_size = lasttrain

trainx = tf.placeholder(tf.float32, [batch_size, reclen])
testx = tf.placeholder(tf.float32, [reclen])
trainl = tf.placeholder(tf.int32, [batch_size])
CDs = tf.reduce_sum(tf.mul(trainx, testx), axis=1)
bestCDs, bestCDIDxs = tf.nn.top_k(CDs, num_ks)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


num_errors = 0
num_steps = 1
for step in range(num_steps):
	for itestitem, testitem in enumerate(testvals):
		cand_ids = []
		if b_use_clusters:
			CDs = [sum([val * testitem[ival] for ival, val in enumerate(centroid)]) for centroid in centroids]
			npCDs = np.asarray(CDs)
			for icluster in range(num_clusters_to_get):
				i_best_centroid = np.argmax(npCDs)
				cand_ids = cand_ids + [i for i in clusters[i_best_centroid]]
				npCDs = np.delete(npCDs, i_best_centroid)
		else:
			cand_ids = [i for i in range(len(trainvals))] # Note! Not actually used


		fd = {trainx: trainvals, testx: testitem}
		bestCDIDxs_vals, = sess.run([bestCDIDxs], feed_dict=fd)

		vote_board = { lval : 0.0 for lval in labels_unique}
		for iidx, idx in enumerate(bestCDIDxs_vals):
			vote_board[labels[idx]] += 1.0/float(iidx+1)

		top_label = 'unfounded'
		best_score = -1.0
		for lval in vote_board:
			if (vote_board[lval] > best_score) :
				best_score = vote_board[lval]
				top_label = lval
		if (top_label != labels[itestitem + lasttrain]):
			num_errors += 1

	print num_errors, ' errors out of ', len(testvals), 'accuracy = ', (len(testvals) - num_errors) * 100.0 / len(testvals), '%'

sess.close()






print 'done'