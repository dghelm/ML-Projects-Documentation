from __future__ import print_function
import numpy as np
import time
import math
import tensorflow as tf
import load_fmri_multich as input_data
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

CHANNELS = 4

RETRAIN = True
MODEL_FILE_DIR = "./models/"
MODEL_FILE = "cae_mch_161114-004512_4999.ckpt"
LOAD_FEATURES = True
CLASSIFY = True
RESET_GLOBAL_STEP = False
NEW_GLOBAL_STEP = 4000


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

# reshaped = np.reshape(trainimgs[0], [56,64,48, CHANNELS])
# reshaped = reshaped [:,:,22:25,0]
#
# imgplot = plt.imshow(reshaped)




total_n      = len(fmri.train.labels) + len(fmri.test.labels)
chan_size = 56*64*48

# Define Network
n1 = 32
n2 = 64
n3 = 128
n4 = 256
# n4 = 64
ksize = 7

with tf.name_scope('weights') as scope:
    weights = {
        'ce1': tf.Variable(tf.random_normal([ksize, ksize, ksize, CHANNELS, n1],  stddev=0.1)),
        'ce2': tf.Variable(tf.random_normal([ksize, ksize, ksize, n1, n2], stddev=0.1)),
        'ce3': tf.Variable(tf.random_normal([ksize, ksize, ksize, n2, n3], stddev=0.1)),
        'ce4': tf.Variable(tf.random_normal([ksize, ksize, ksize, n3, n4], stddev=0.1)),
        'cd4': tf.Variable(tf.random_normal([ksize, ksize, ksize, n3, n4], stddev=0.1)),
        'cd3': tf.Variable(tf.random_normal([ksize, ksize, ksize, n2, n3], stddev=0.1)),
        'cd2': tf.Variable(tf.random_normal([ksize, ksize, ksize, n1, n2], stddev=0.1)),
        'cd1': tf.Variable(tf.random_normal([ksize, ksize, ksize, CHANNELS, n1],  stddev=0.1))
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
        # 'bd1': tf.Variable(tf.random_normal([1],  stddev=0.1))
        'bd1': tf.Variable(tf.random_normal([CHANNELS],  stddev=0.1))
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
            , tf.pack([tf.shape(_X)[0], 56, 64, 48, CHANNELS]), strides=[1, 2, 2, 2, 1]
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
    tf.scalar_summary('dropout_keep_probability', keepprob, collections=['CAE'])
# keepprob = tf.placeholder(tf.float32, name="dropout_prob")


# Prediction of whole net for what output is.
# We're trying to mininimize its distance from input x.
pred = cae(x, weights, biases, keepprob)['out']

# Calculate the cost function for outputs.
# Here using sum of the squared errors.
with tf.name_scope('SSE_CAE') as scope:
    #diff = cae(x, weights, biases, keepprob)['out'] - tf.reshape(y, shape=[-1, 56, 64, 48, 1],)
    self_diff = pred - tf.reshape(x, shape=[-1, 56, 64, 48, CHANNELS],)
    with tf.name_scope('total'):
        cost = tf.reduce_sum(tf.square(self_diff))
    tf.scalar_summary('SSE_CAE', cost, collections=['CAE'])


# Learning rate.
# learning_rate = 0.0001
global_step = tf.Variable(0, trainable=False)

## with 4 batches per epoch, global step is about 4x the epoch
## initial * decay^(globalstep / decaystep)
with tf.name_scope('learning_rate') as scope:
    learning_rate =  tf.train.exponential_decay(0.00015, global_step, 2000, 0.8)
    tf.scalar_summary('learning_rate', learning_rate, collections=['CAE'])


# Training of weights function specified here using Adam Optimizer
## TODO: Better understand what Adam is.
with tf.name_scope('train') as scope:
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)


with tf.name_scope('reconstruction_preview'):
    inp = tf.reshape(x, [56,64,48,CHANNELS])
    outp = tf.reshape(pred, [56,64,48,CHANNELS])
    for i in range(CHANNELS):
        inp_img = tf.reshape( inp[:,:,23:26,i]*255 , [1, 56, 64, 3] )
        outp_img = tf.reshape(outp[:,:,23:26,i]*255, [1, 56, 64, 3] )
        tf.image_summary("Ch " + str(i) + " Input Slice", inp_img, 50, collections=["CAE_img"])
        tf.image_summary("Ch " + str(i) + " Output Slice", outp_img, 50, collections=["CAE_img"])



# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.elu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('nn_weights'):
            nn_weights = weight_variable([input_dim, output_dim])
            variable_summaries(nn_weights, layer_name + '/weights')
        with tf.name_scope('nn_biases'):
            nn_biases = bias_variable([output_dim])
            variable_summaries(nn_biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, nn_weights) + nn_biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations





y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

with tf.name_scope("encoded_reshape"):
   encoded_reshape = tf.reshape(cae(x, weights, biases, 1)['ce4'], shape=[-1, 4*4*3*n4])

nn_keep_prob = tf.placeholder(tf.float32)

hidden1 = nn_layer(encoded_reshape, (4*4*3*n4), (4*3*3*n2),'layer1')
with tf.name_scope('l1_fully_dropout'):
    tf.scalar_summary('l1_fully_dropout_keep_probability', nn_keep_prob)
    l1_fully_dropped = tf.nn.dropout(hidden1, nn_keep_prob)

hidden2 = nn_layer(l1_fully_dropped, 4*3*3*n2, 4*3*3, 'layer2')
with tf.name_scope('l2_fully_dropout'):
    tf.scalar_summary('l2_fully_dropout_keep_probability', nn_keep_prob)
    l2_fully_dropped = tf.nn.dropout(hidden2, nn_keep_prob)

  # Do not apply softmax activation yet, see below.
nn_y = nn_layer(l2_fully_dropped, 4*3*3, 2, 'layer3', act=tf.identity)



## with ? batches per epoch, nn global step is about ?x the epoch
## initial * decay^(globalstep / decaystep)
nn_global_step = tf.Variable(0, trainable=False)

with tf.name_scope('nn_learning_rate') as scope:
    nn_learning_rate =  tf.train.exponential_decay(0.00001, nn_global_step, 110, 1.)
    tf.scalar_summary('nn_learning_rate', nn_learning_rate)

##TODO: Better understand cross entropy and softmax version
with tf.name_scope('nn_cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    nn_diff = tf.nn.softmax_cross_entropy_with_logits(nn_y,y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(nn_diff)
    tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('nn_train'):
    train_step = tf.train.AdamOptimizer(nn_learning_rate).minimize(
        cross_entropy, global_step=nn_global_step)

with tf.name_scope('nn_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(nn_y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

#with tf.name_Scope('cost_to_mean'):
#   n = tf.shape_n(x)[0]
#    expanded_mean = tf.expand_dims( tf.reduce_mean(x,0), 0 )
#    expanded_mean = tf.pad( expanded_mean, [[0,n-1], [0,0], [0,0], [0,0],[0,0]])
#    mean_diff = expanded_mean - tf.reshape(x, shape=[-1, 56, 64, 48, CHANNELS],)
#    cost_to_mean = tf.reduce_sum(tf.square(self_diff))
#    tf.scalar_summary('Cost_to_Mean', cost_to_mean, collections=['CAE'])

# Initialize all vars
print ("Functions ready")

## TODO: Determine if I need this or .gloval_variables_initializer()
init = tf.initialize_all_variables()
saver = tf.train.Saver()


with tf.Session() as sess:

    merged = tf.merge_all_summaries(key='CAE')
    img_merged = tf.merge_all_summaries(key='CAE_img')
    timestr = time.strftime("%y%m%d-%H%M%S")
    train_writer = tf.train.SummaryWriter('./tensorflow_logs/train_32-256_mch_'+timestr+'/', sess.graph)
    test_writer = tf.train.SummaryWriter('./tensorflow_logs/test_32-256_mch_'+timestr+'/', sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)

    # batch_size = 16
    batch_size = len(testlabels)
    n_epochs   = 3000
    epoch_i = 0



    if LOAD_FEATURES:
        # Load weights!
        saver.restore(sess, MODEL_FILE_DIR + MODEL_FILE)
        print("Model Restored.")

    if RESET_GLOBAL_STEP:

        print(global_step.eval())

        sess.run( tf.assign(global_step, NEW_GLOBAL_STEP ) )

        print(global_step.eval())

    if RETRAIN:
        print("Start CAE training...")

        for epoch_i in range(n_epochs):
            for batch_i in range(fmri.train.num_examples // batch_size):
                batch_xs, batch_labels = fmri.train.next_batch(batch_size)
                # trainbatch = np.array([img for img in batch_xs])
                sess.run(optm, feed_dict={x: batch_xs
                    , y: batch_xs, keepprob: 0.7})

            summary, train_cost = sess.run([merged, cost], feed_dict={x: batch_xs, y: batch_xs, keepprob: 1.})
            train_writer.add_summary(summary, epoch_i)
            print ("[%02d/%02d] cost: %.4f lr: %.7f gs: %f" %
                   (epoch_i, n_epochs, train_cost, learning_rate.eval(), global_step.eval(), ))
            if (epoch_i % 3) == 0:
                test_xs, _ = fmri.test.next_batch(batch_size)
                summary, test_cost = sess.run([merged, cost], feed_dict={x: test_xs, y: test_xs, keepprob: 1.})
                test_writer.add_summary(summary, epoch_i)
                print("Test cost: %.4f" % (test_cost))
            if (epoch_i < 10) or (epoch_i % 150 == 0):
                test_xs, _ = fmri.test.next_batch(batch_size)
                test_xs = np.reshape(test_xs[0], [1, -1] )
                summary = sess.run(img_merged, feed_dict={x:test_xs, keepprob: 1.})
                test_writer.add_summary(summary, epoch_i)

            if (epoch_i % 500) == 0:
                save_path = saver.save(sess, MODEL_FILE_DIR + "cae_mch_"+ timestr +"_"+ str(epoch_i) +".ckpt")

        print("CAE Training done. Saving Model...")
        save_path = saver.save(sess, MODEL_FILE_DIR + "cae_mch_"+ timestr +"_"+ str(epoch_i) +".ckpt")
        print("Model Saved in file: %s" % save_path)
    else:
        # Load weights!
        saver.restore(sess, MODEL_FILE_DIR + MODEL_FILE)
        print("Model Restored.")

    test_xs, _ = fmri.test.next_batch(batch_size)
    summary, test_cost = sess.run([merged, cost], feed_dict={x: test_xs, y: test_xs, keepprob: 1.})
    test_writer.add_summary(summary, epoch_i)
    print("Test cost: %.4f" % (test_cost))

    test_xs = np.reshape(test_xs[0], [1, -1] )
    # test_xs = tf.expand_dims(test_xs[0], 0 )
    summary = sess.run(img_merged, feed_dict={x:test_xs, keepprob: 1.})
    test_writer.add_summary(summary, epoch_i)


    if CLASSIFY:
        nn_batch_size = batch_size
        nn_n_epochs   = 2000

        merged = tf.merge_all_summaries()

        print("Start Classifier NN training...")
        for epoch_i in range(nn_n_epochs):
            if epoch_i % 5 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, accuracy], feed_dict={x: testimgs, y_: testlabels, nn_keep_prob:1.})
                test_writer.add_summary(summary, epoch_i)
                print('Accuracy at epoch %s: %s' % (epoch_i, acc))

            for batch_i in range(fmri.train.num_examples // batch_size):
                xs, ys = fmri.train.next_batch(nn_batch_size)
                summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys, nn_keep_prob:0.7})
                train_writer.add_summary(summary, epoch_i)

            if (epoch_i % 500) == 0:
                save_path = saver.save(sess, MODEL_FILE_DIR + "cae_mch_nn_"+ timestr +"_"+ str(epoch_i) +".ckpt")


    train_writer.close()
    test_writer.close()
    sess.close()
    print ("Session closed.")
