import numpy as np
import tensorflow as tf
import csv
import random
import math

num_centroids = 2
num_cluster_iters = 2
num_clusters_to_get = 2
num_ks = 200


knn_in_fname = '/devlink2/data/cifar/imgnet_small.txt'
csvfile = open(knn_in_fname, 'rt')
reader = csv.reader(csvfile, delimiter=',')
b_use_clusters = False

allrows = []
labels = []
for row in reader:
	labels.append(row[0])
	vals = [float(d) for d in row[1:]]
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

batch_size = num_ks

kx = tf.placeholder(tf.float32, [batch_size, reclen])
x = tf.placeholder(tf.float32, [reclen])
W = tf.Variable(tf.random_uniform([reclen, reclen], 0.3))
b = tf.Variable(tf.random_uniform([reclen], 0.3))
y = tf.matmul(kx, tf.clip_by_value(W, 0.0, 10.0)) # + b
# CDs = tf.reduce_sum(tf.mul(kx, x), axis=1)
asqrt = tf.sqrt(tf.reduce_sum(tf.mul(tf.square(x), y)))
bsqrt = tf.sqrt(tf.reduce_sum(tf.mul(tf.square(kx), y), axis=1))
CDs = tf.divide(tf.reduce_sum(tf.mul(kx, tf.mul(x, y)), axis=1), tf.mul(asqrt, bsqrt))
# the following is 0.0 for a match and 1.0/(1+rank) for an error
match_factors = tf.placeholder(tf.float32, [batch_size])
losses = tf.divide(match_factors, tf.subtract(1.0, CDs))
loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.5).minimize(loss)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


num_errors = 0
num_steps = 10
for step in range(num_steps+1):
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
			cand_ids = [i for i in range(len(trainvals))]


		CD_pairs = [(sum([val * testitem[ival] for ival, val in enumerate(trainvals[icand])]), icand) for icand in cand_ids]
		CD_pairs = sorted(CD_pairs, key=lambda tup: tup[0], reverse=True)
		del CD_pairs[num_ks:]
		kx_vals = [trainvals[tup[1]] for tup in CD_pairs]
		CDvals = [tup[0] for tup in CD_pairs]
		loss_factors = [(1.0/(rank+1)) if labels[tup[1]] != labels[itestitem + lasttrain] else 0.0 for rank, tup in enumerate(CD_pairs)]
		fd = {x: testitem, kx: kx_vals, match_factors: loss_factors}
		lossval1 = sess.run([loss], feed_dict=fd)
		sess.run([train_step], feed_dict=fd)
		lossval2 = sess.run([loss], feed_dict=fd)
		print 'step:', step, ', loss change: ', lossval1, lossval2
		vote_board = { lval : 0.0 for lval in labels_unique}
		for ipair, cd_pair in enumerate(CD_pairs):
			if ipair > num_ks:
				break
			vote_board[labels[cd_pair[1]]] += 1.0/float(ipair+1)

		top_label = 'unfounded'
		best_score = -1.0
		for lval in vote_board:
			if (vote_board[lval] > best_score) :
				best_score = vote_board[lval]
				top_label = lval
		if (top_label != labels[itestitem + lasttrain]):
			num_errors += 1

	print num_errors, ' errors out of ', len(testvals)

sess.close()






print 'done'