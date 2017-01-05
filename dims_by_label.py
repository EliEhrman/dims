import numpy as np
import tensorflow as tf
import csv
import random
import matplotlib.pyplot as plt

# knn_large_fname = '/devlink/data/cifar/cifar.txt'
# knn_small_fname = '/devlink/data/cifar/cifar_small.txt'
knn_large_fname = '/devlink2/data/imagenet/llayer.txt'
knn_small_fname = '/devlink2/data/imagenet/imgnet_small.txt'
b_fname_included = True
same_factor = 1.0 # reduce distance when labels the same by this factor
different_factor = 1.0
num_iter_steps = 100000
batch_size = 50
output_size = 10 # how many properties to generate


csvfile = open(knn_large_fname, 'rt')
reader = csv.reader(csvfile, delimiter=',')

allrows = []
labels = []
fnames = []
data_start_col = 1
if b_fname_included:
	data_start_col = 2

for row in reader:
	labels.append(row[0])
	fnames.append(row[1])
	vals = [float(d) for d in row[data_start_col:]]
	allrows.append(vals)

csvfile.close()

allcols = zip(*allrows)
mins = [min(col) for col in allcols]
maxs = [max(col) for col in allcols]
allvals = []
for row in allrows:
	vals = [(col - mins[icol]) / (maxs[icol] - mins[icol]) if (maxs[icol] - mins[icol]) > 0.0 else 0.0 for icol, col in enumerate(row)]
	sqrt = sum([x ** 2 for x in vals])
	allvals.append([x / sqrt for x in vals])
# print mins, maxs

labels_unique = set(labels)
labels_map = {label:il for il, label in enumerate(labels_unique)}
labelnums = [labels_map[label] for label in labels]
num_unique_labels = len(labels_unique)
rsize = batch_size ** 2
reclen = len(allrows[0])
numrecs = len(allrows)
rfirst = tf.placeholder(tf.int32)
rsecond = tf.placeholder(tf.int32)
r1arr = tf.placeholder(tf.int32, [rsize])
r2arr = tf.placeholder(tf.int32, [rsize])

x = tf.placeholder(tf.float32, [batch_size, reclen])
tlabelnums = tf.placeholder(tf.int32, batch_size)
W = tf.Variable(tf.random_uniform([reclen, output_size], 0.3))
b = tf.Variable(tf.random_uniform([output_size], 0.3))
x1 = tf.gather(x, r1arr)
x2 = tf.gather(x, r2arr)
l1 = tf.gather(labelnums, r1arr)
l2 = tf.gather(labelnums, r2arr)
y = tf.matmul(x, tf.clip_by_value(W, 0.0, 10.0)) # + b
y1 = tf.gather(y, r1arr)
y2 = tf.gather(y, r2arr)
xlist = tf.unstack(x)
ylist = tf.unstack(y)
# dist1 = tf.reduce_sum(tf.squared_difference(x1, x2), axis=1)
# dist2 = tf.reduce_sum(tf.squared_difference(y1, y2), axis=1)
dist1 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
# dist1eq = tf.mul(tf.reduce_sum(tf.multiply(x1, x2), axis=1), 0.1)
# dist1neq = tf.mul(tf.reduce_sum(tf.multiply(x1, x2), axis=1), 10.0)
# tsame = tf.select(tf.equal(l1, l2), dist1eq, dist1neq)
# There are far fewer examples of l1 and l1 equal than not equal,
# To balanace this, we create weights that will give more emphasis to
# the equal cases
dist2 = tf.reduce_sum(tf.multiply(y1, y2), axis=1)
err_eq = tf.mul((tf.mul(dist1, same_factor) - dist2) ** 2, float(num_unique_labels))
err_neq = (tf.mul(dist1, different_factor) - dist2) ** 2
err = tf.reduce_sum(tf.select(tf.equal(l1, l2), err_eq, err_neq))
train_step = tf.train.AdagradOptimizer(0.05).minimize(err)
# dist = tf.reduce_sum(tf.squared_difference(x1, x2))

TheMatrix = np.empty([reclen, output_size], dtype=np.float64)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plotvec = []
plt.figure('dims by label')
plt.yscale('log')
plt.ion()
for step in range(num_iter_steps+1):
	r1 = [random.randint(0, batch_size-1) for i in range(rsize)]
	r2 = [random.randint(0, batch_size-1) for i in range(rsize)]
	batch_start = random.randint(0, numrecs - batch_size)
	fd = {	x: allvals[batch_start : batch_start+batch_size], r1arr: r1, r2arr: r2}
	# l1vals, l2vals, tsamevals = sess.run([l1, l2, err], feed_dict=fd)
	if step is 0:
		print 'W\n', sess.run([W], feed_dict=fd)
	# errval2 = sess.run([err], feed_dict=fd)
	if step % (num_iter_steps / 100) is 0:
		errval1 = sess.run([err], feed_dict=fd)
		print 'step:', step, ', err : ', errval1
		plotvec.append(errval1)
		plt.plot(plotvec)
		plt.pause(0.05)
	sess.run([train_step], feed_dict=fd)

	if step == num_iter_steps:
		# print 'W\n', sess.run([W], feed_dict=fd)
		TheMatrix = sess.run(W, feed_dict=fd)
		print TheMatrix

sess.close()

TheMatrix = np.clip(TheMatrix, 0, 10.0)
print TheMatrix
output = np.dot(allvals, TheMatrix )
csvfile = open(knn_small_fname, 'wt')
writer = csv.writer(csvfile, delimiter=',')
for irow, row in enumerate(output):
	data = [str(d) for d in row]
	label_prefix = [str(labels[irow])]
	if b_fname_included:
		label_prefix.append(str(fnames[irow]))
	writer.writerow(label_prefix + data)
csvfile.close()

print 'done'

while True: plt.pause(0.5)

# for step in range(10):
# 	# fd = {	x: allvals[0:batch_size], x1: allvals[random.randint(1, batch_size)],
# 	# 		x2: allvals[random.randint(1, batch_size)]}
# 	r1 = random.randint(0, batch_size-1)
# 	r2 = random.randint(0, batch_size-1)
# 	fd = {	x: allvals[0:batch_size], rfirst: r1, rsecond: r2}
# 	# onex_val = sess.run(x1, feed_dict=fd)
# 	# x1, x2, y1, y2, distval = sess.run([x[rfirst], x[rsecond], y[rfirst], y[rsecond], err], feed_dict=fd)
# 	print 'err change: ', errval1, errval2
# xv, x1v = sess.run([x, x1], feed_dict=fd)
# print 'err change: ', xv, x1v
#
# print 'done'




