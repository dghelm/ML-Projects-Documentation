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
n4 = 128
ksize_a = 5
ksize = 5
with tf.name_scope('weights') as scope:
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
with tf.name_scope('biases') as scope:
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
    with tf.name_scope('input_reshape') as scope:
        _input_r = tf.reshape(_X, shape=[-1, 56, 64, 48, CHANNELS])
    # Encoder
    with tf.name_scope('encoders') as scope:
        _ce1 = tf.nn.sigmoid(tf.add(tf.nn.conv3d(_input_r, _W['ce1']
            , strides=[1, 2, 2, 2, 1], padding='SAME'), _b['be1']), )
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
    with tf.name_scope('decoders') as scope:
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
    with tf.name_scope('output') as scope:
        _out = _cd1
    return {'input_r': _input_r, 'ce1': _ce1, 'ce2': _ce2, 'ce3': _ce3, 'ce4': _ce4
        , 'cd4': _cd4 , 'cd3': _cd3, 'cd2': _cd2, 'cd1': _cd1
        , 'layers': (_input_r, _ce1, _ce2, _ce3, _ce4, _cd4, _cd3, _cd2, _cd1)
        , 'out': _out}
print ("Network ready")

# Create Placeholders for input and output.
## TODO: Do these need to be different? Can we half our memory by not loading duplicate data into these placeholders?

x = tf.placeholder(tf.float32, [None, dim], name="input")
y = tf.placeholder(tf.float32, [None, dim], name="flat_output")

# Keep prob is the probability used for drop out.
# Placeholder needed for training vs testing.
with tf.name_scope('dropout'):
    keepprob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keepprob)
# keepprob = tf.placeholder(tf.float32, name="dropout_prob")


# Prediction of whole net for what output is.
# We're trying to mininimize its distance from input x.
pred = cae(x, weights, biases, keepprob)['out']

# Calculate the cost function for outputs.
# Here using sum of the squared errors.
with tf.name_scope('SSE_CAE') as scope:
    #diff = cae(x, weights, biases, keepprob)['out'] - tf.reshape(y, shape=[-1, 56, 64, 48, 1],)
    diff = pred - tf.reshape(x, shape=[-1, 56, 64, 48, 1],)
    with tf.name_scope('total'):
        cost = tf.reduce_sum(tf.square(diff))
    tf.scalar_summary('SSE_CAE', cost)



# Learning rate.
## TODO: Why is this a constant set here?
# learning_rate = 0.0001
global_step = tf.Variable(0, trainable=False)

## with 4 batches per epoch, global step is about 4x the epoch
## initial * decay^(globalstep / decaystep)
with tf.name_scope('learning_rate') as scope:
    learning_rate =  tf.train.exponential_decay(0.00013, global_step, 110, 0.75)
    tf.scalar_summary('learning_rate', learning_rate)

# Training of weights function specified here using Adam Optimizer
## TODO: Better understand what Adam is.
with tf.name_scope('train') as scope:
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)









#
#
#
# # We can't initialize these variables to 0 - the network will get stuck.
# def weight_variable(shape):
#     """Create a weight variable with appropriate initialization."""
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     """Create a bias variable with appropriate initialization."""
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# def variable_summaries(var, name):
#     """Attach a lot of summaries to a Tensor."""
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.scalar_summary('mean/' + name, mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.scalar_summary('stddev/' + name, stddev)
#         tf.scalar_summary('max/' + name, tf.reduce_max(var))
#         tf.scalar_summary('min/' + name, tf.reduce_min(var))
#         tf.histogram_summary(name, var)
#
# def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
#     """Reusable code for making a simple neural net layer.
#     It does a matrix multiply, bias add, and then uses relu to nonlinearize.
#     It also sets up name scoping so that the resultant graph is easy to read,
#     and adds a number of summary ops.
#     """
#     # Adding a name scope ensures logical grouping of the layers in the graph.
#     with tf.name_scope(layer_name):
#         # This Variable will hold the state of the weights for the layer
#         with tf.name_scope('weights'):
#             weights = weight_variable([input_dim, output_dim])
#             variable_summaries(weights, layer_name + '/weights')
#         with tf.name_scope('biases'):
#             biases = bias_variable([output_dim])
#             variable_summaries(biases, layer_name + '/biases')
#         with tf.name_scope('Wx_plus_b'):
#             preactivate = tf.matmul(input_tensor, weights) + biases
#             tf.histogram_summary(layer_name + '/pre_activations', preactivate)
#         activations = act(preactivate, name='activation')
#         tf.histogram_summary(layer_name + '/activations', activations)
#         return activations
#
# with tf.name_scope("encoded_reshape"):
#    encoded_reshape = tf.reshape(encoded, )
#
# hidden1 = nn_layer(encoded, (4*4*3*128), (4*3*3*32),'layer1')
# with tf.name_scope('l1_fully_dropout'):
#     l1_fully_keep_prob = tf.placeholder(tf.float32)
#     tf.scalar_summary('l1_fully_dropout_keep_probability', l1_fully_keep_prob)
#     l1_fully_dropped = tf.nn.dropout(hidden1, l1_fully_keep_prob)
#
# hidden2 = nn_layer(l1_fully_dropped, 4*3*3*32, 4*3*3, 'layer2')
# with tf.name_scope('l2_fully_dropout'):
#     l2_fully_keep_prob = tf.placeholder(tf.float32)
#     tf.scalar_summary('l2_fully_dropout_keep_probability', l2_fully_keep_prob)
#     l2_fully_dropped = tf.nn.dropout(hidden2, l2_fully_keep_prob)
#
#   # Do not apply softmax activation yet, see below.
# fully_y = nn_layer(l2_fully_dropped, 4*3*3, 2, 'layer3', act=tf.identity)



# Initialize all vars
print ("Functions ready")




## TODO: Should this be interactive session for this framework?
with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter('./tensorflow_logs', sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)

    # mean_img = np.mean(mnist.train.images, axis=0)
    #mean_img = np.zeros((149760))
    # Fit all training data
    batch_size = 128/2
    n_epochs   = 250

    print("Start alff training..")
    for epoch_i in range(n_epochs):
        for batch_i in range(fmri.train.num_examples // batch_size):
            batch_xs, _ = fmri.train.next_batch(batch_size)
            trainbatch = np.array([img for img in batch_xs])
            sess.run(optm, feed_dict={x: trainbatch
                                      , y: trainbatch, keepprob: 0.5})

        summary, train_cost = sess.run([merged, cost], feed_dict={x: trainbatch, y: trainbatch, keepprob: 1.})
        train_writer.add_summary(summary, epoch_i)
        print ("[%02d/%02d] cost: %.4f lr: %.7f gs: %f" %
               (epoch_i, n_epochs, train_cost, learning_rate.eval(), global_step.eval(), ))
        #print(tf.Tensor.eval([global_step, learning_rate]))
        # print ("[%02d/%02d] cost: %.4f  lr: %.6f" % (epoch_i, n_epochs, train_cost, learning_rate ))
    print("Alff Training done.")


    test_xs, _ = fmri.test.next_batch(testlabels.shape[0])
    test_xs_norm = np.array([img for img in test_xs])
    #test_xs_norm = np.array([img - mean_img for img in test_xs])
    #recon = sess.run(pred, feed_dict={x: test_xs_norm, keepprob: 1.})
    test_cost = sess.run(cost, feed_dict={x: test_xs, y: test_xs, keepprob: 1.})
    print("Tests Cost: %f " % (test_cost))

    sample_output = sess.run(cae(x, weights, biases, keepprob)['ce4'], feed_dict={x:test_xs[:2], keepprob:1.})
    print(sample_output[0,0,:,:,0])


    layers = sess.run(cae(x, weights, biases, keepprob)['layers']
                      , feed_dict={x: test_xs_norm, keepprob: 1.})


    for i in range(len(layers)):
        currl = layers[i]
        print (("Shape of layer %d is %s") % (i+1, currl.shape,))
        #if i == 6:
            #print("output of layer is")
            #print(currl[0,:,:,:,0])

    train_writer.close()
    sess.close()
    print ("Session closed.")