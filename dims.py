import numpy as np
import tensorflow as tf
import csv
import random
import matplotlib.pyplot as plt

# knn_large_fname = '/devlink/data/cifar/cifar.txt'
# knn_small_fname = '/devlink/data/cifar/cifar_small.txt'
knn_large_fname = '/devlink2/data/imagenet/llayer.txt'
knn_small_fname = '/devlink2/data/imagenet/imgnet_small.txt'
# knn_large_fname = '/devlink/data/metamath/setexpanded.txt'
# knn_small_fname = '/devlink/data/metamath/setshrunk.txt'

b_fname_included = True
num_steps = 1000000
batch_size = 30
output_size = 40

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

rsize = batch_size ** 2
reclen = len(allrows[0])
numrecs = len(allrows)
rfirst = tf.placeholder(tf.int32)
rsecond = tf.placeholder(tf.int32)
r1arr = tf.placeholder(tf.int32, [rsize])
r2arr = tf.placeholder(tf.int32, [rsize])

x = tf.placeholder(tf.float32, [batch_size, reclen])
W = tf.Variable(tf.random_uniform([reclen, output_size], 0.3))
b = tf.Variable(tf.random_uniform([output_size], 0.3))
x1 = tf.gather(x, r1arr)
x2 = tf.gather(x, r2arr)
y = tf.matmul(x, tf.clip_by_value(W, 0.0, 10.0)) # + b
y1 = tf.gather(y, r1arr)
y2 = tf.gather(y, r2arr)
xlist = tf.unstack(x)
ylist = tf.unstack(y)
# dist1 = tf.reduce_sum(tf.squared_difference(x1, x2), axis=1)
# dist2 = tf.reduce_sum(tf.squared_difference(y1, y2), axis=1)
dist1 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
dist2 = tf.reduce_sum(tf.multiply(y1, y2), axis=1)
err = tf.reduce_mean((dist1 - dist2) ** 2)
train_step = tf.train.AdagradOptimizer(0.05).minimize(err)
# dist = tf.reduce_sum(tf.squared_difference(x1, x2))

TheMatrix = np.empty([reclen, output_size], dtype=np.float64)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plotvec = []
plt.figure('dims - the one and only!')
plt.yscale('log')
plt.ion()
for step in range(num_steps+1):
	r1 = [random.randint(0, batch_size-1) for i in range(rsize)]
	r2 = [random.randint(0, batch_size-1) for i in range(rsize)]
	batch_start = random.randint(0, numrecs - batch_size)
	fd = {	x: allvals[batch_start : batch_start+batch_size], r1arr: r1, r2arr: r2}
	if step is 0:
		print 'W\n', sess.run([W], feed_dict=fd)
	if step % (num_steps / 100) is 0:
		errval1 = sess.run([err], feed_dict=fd)
		print 'step:', step, ', err : ', errval1
		plotvec.append(errval1)
		plt.plot(plotvec)
		plt.pause(0.05)

	sess.run([train_step], feed_dict=fd)

	if step == num_steps:
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




