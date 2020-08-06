from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
import load_fmri as input_data

chans = ["alff"]
CHANNELS = 1

print ("Packages loaded")

# Load Data
fmri = input_data.read_data_sets("./data/AllSubjects4cat.hdf5", fraction=1, channels=CHANNELS)
trainimgs   = fmri.train.images
trainlabels = fmri.train.labels
testimgs    = fmri.test.images
testlabels  = fmri.test.labels
ntrain      = trainimgs.shape[0]
ntest       = testimgs.shape[0]
dim         = trainimgs.shape[1]
nout        = trainlabels.shape[1]


# Define Network
## TODO: If saved weight file exists, use it!
n1 = 16
n2 = 32
n3 = 64
n4 = 64
ksize_a = 5
ksize = 5
weights = {
    'ce1': tf.Variable(tf.random_normal([ksize_a, ksize_a, ksize_a, CHANNELS, n1],  stddev=0.1)),
    'ce2': tf.Variable(tf.random_normal([ksize, ksize, ksize, n1, n2], stddev=0.1)),
    'ce3': tf.Variable(tf.random_normal([ksize, ksize, ksize, n2, n3], stddev=0.1)),
    'ce4': tf.Variable(tf.random_normal([ksize, ksize, ksize, n3, n4], stddev=0.1)),
    'cd4': tf.Variable(tf.random_normal([ksize, ksize, ksize, n3, n4], stddev=0.1)),
    'cd3': tf.Variable(tf.random_normal([ksize, ksize, ksize, n2, n3], stddev=0.1)),
    'cd2': tf.Variable(tf.random_normal([ksize, ksize, ksize, n1, n2], stddev=0.1)),
    'cd1': tf.Variable(tf.random_normal([ksize_a, ksize_a, ksize_a, CHANNELS, n1],  stddev=0.1))
}
biases = {
    'be1': tf.Variable(tf.random_normal([n1], stddev=0.1)),
    'be2': tf.Variable(tf.random_normal([n2], stddev=0.1)),
    'be3': tf.Variable(tf.random_normal([n3], stddev=0.1)),
    'be4': tf.Variable(tf.random_normal([n4], stddev=0.1)),
    'bd4': tf.Variable(tf.random_normal([n3], stddev=0.1)),
    'bd3': tf.Variable(tf.random_normal([n2], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n1], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1],  stddev=0.1))
}


# Define our CAE:

def cae(_X, _W, _b, _keepprob):
    _input_r = tf.reshape(_X, shape=[-1, 56, 64, 48, CHANNELS])
    # Encoder
    _ce1 = tf.nn.sigmoid(tf.add(tf.nn.conv3d(_input_r, _W['ce1']
        , strides=[1, 2, 2, 2, 1], padding='SAME'), _b['be1']))
    _ce1 = tf.nn.dropout(_ce1, _keepprob)
    _ce2 = tf.nn.sigmoid(tf.add(tf.nn.conv3d(_ce1, _W['ce2']
        , strides=[1, 2, 2, 2, 1], padding='SAME'), _b['be2']))
    _ce2 = tf.nn.dropout(_ce2, _keepprob)
    _ce3 = tf.nn.sigmoid(tf.add(tf.nn.conv3d(_ce2, _W['ce3']
        , strides=[1, 2, 2, 2, 1], padding='SAME'), _b['be3']))
    _ce3 = tf.nn.dropout(_ce3, _keepprob)
    _ce4 = tf.nn.sigmoid(tf.add(tf.nn.conv3d(_ce3, _W['ce4']
         , strides=[1, 2, 2, 2, 1], padding='SAME'), _b['be4']))
    _ce4 = tf.nn.dropout(_ce4, _keepprob)
    # Decoder
    _cd4 = tf.nn.sigmoid(tf.add(tf.nn.conv3d_transpose(_ce4, _W['cd4']
       , tf.pack([tf.shape(_X)[0], 7, 8, 6, n3]), strides=[1, 2, 2, 2, 1]
       , padding='SAME'), _b['bd4']))
    _cd4 = tf.nn.dropout(_cd4, _keepprob)
    _cd3 = tf.nn.sigmoid(tf.add(tf.nn.conv3d_transpose(_cd4, _W['cd3']
        , tf.pack([tf.shape(_X)[0], 14, 16, 12, n2]), strides=[1, 2, 2, 2, 1]
        , padding='SAME'), _b['bd3']))
    _cd3 = tf.nn.dropout(_cd3, _keepprob)
    _cd2 = tf.nn.sigmoid(tf.add(tf.nn.conv3d_transpose(_cd3, _W['cd2']
        , tf.pack([tf.shape(_X)[0], 28, 32, 24, n1]), strides=[1, 2, 2, 2, 1]
        , padding='SAME') , _b['bd2']))
    _cd2 = tf.nn.dropout(_cd2, _keepprob)
    _cd1 = tf.nn.sigmoid(tf.add(tf.nn.conv3d_transpose(_cd2, _W['cd1']
        , tf.pack([tf.shape(_X)[0], 56, 64, 48, 1]), strides=[1, 2, 2, 2, 1]
        , padding='SAME'), _b['bd1']))
    _cd1 = tf.nn.dropout(_cd1, _keepprob)
    _out = _cd1
    return {'input_r': _input_r, 'ce1': _ce1, 'ce2': _ce2, 'ce3': _ce3, 'ce4': _ce4
        , 'cd4': _cd4 , 'cd3': _cd3, 'cd2': _cd2, 'cd1': _cd1
        , 'layers': (_input_r, _ce1, _ce2, _ce3, _ce4, _cd4, _cd3, _cd2, _cd1)
        , 'out': _out}
print ("Network ready")

# Create Placeholders for input and output.
## TODO: Do these need to be different? Can we half our memory by not loading duplicate data into these placeholders?

x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, dim])

# Keep prob is the probability used for drop out.
# Placeholder needed for training vs testing.

keepprob = tf.placeholder(tf.float32)


# Prediction of whole net for what output is.
# We're trying to mininimize its distance from input x.
pred = cae(x, weights, biases, keepprob)['out']

# Calculate the cost function for outputs.
# Here using sum of the squared errors.
cost = tf.reduce_sum(tf.square(cae(x, weights, biases, keepprob)['out']
            - tf.reshape(y, shape=[-1, 56, 64, 48, 1])))

# Learning rate.
## TODO: Why is this a constant set here?
learning_rate = 0.001

# Training of weights function specified here using Adam Optimizer
## TODO: Better understand what Adam is.
optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize all vars
init = tf.initialize_all_variables()
print ("Functions ready")








## TODO: Should this be interactive session for this framework?
sess = tf.Session()
sess.run(init)
# mean_img = np.mean(mnist.train.images, axis=0)
#mean_img = np.zeros((149760))
# Fit all training data
batch_size = 128/2
n_epochs   = 10

print("Strart training..")
for epoch_i in range(n_epochs):
    for batch_i in range(fmri.train.num_examples // batch_size):
        batch_xs, _ = fmri.train.next_batch(batch_size)
        #trainbatch = np.array([img - mean_img for img in batch_xs])
        trainbatch = np.array([img for img in batch_xs])
        print(".", end="")
        #trainbatch_noisy = trainbatch + 0.3*np.random.randn(
            #trainbatch.shape[0], 784)
        sess.run(optm, feed_dict={x: trainbatch
                                  , y: trainbatch, keepprob: 0.7})
    print ("[%02d/%02d] cost: %.4f" % (epoch_i, n_epochs
        , sess.run(cost, feed_dict={x: trainbatch
                                    , y: trainbatch, keepprob: 1.})))
print("Training done. ")




test_xs, _ = fmri.test.next_batch(batch_size)
test_xs_norm = np.array([img for img in test_xs])
#test_xs_norm = np.array([img - mean_img for img in test_xs])
#recon = sess.run(pred, feed_dict={x: test_xs_norm, keepprob: 1.})
test_cost = sess.run(cost, feed_dict={x: test_xs, y: test_xs, keepprob: 1.})
print(test_cost)

layers = sess.run(cae(x, weights, biases, keepprob)['layers']
                  , feed_dict={x: test_xs_norm, keepprob: 1.})

for i in range(len(layers)):
    currl = layers[i]
    print (("Shape of layer %d is %s") % (i+1, currl.shape,))



sess.close()
print ("Session closed.")