"""Functions for reading and initializing fmri data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import collections
import numpy as np
import h5py
#from six.moves import xrange  # pylint: disable=redefined-builtin

#from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def print(s, end='\n', file=sys.stdout):
    file.write(s + end)
    file.flush()
    
def group_cat(i):
    if int(i) == 1:
        return np.array([1.0,0.0])
    else:
        return np.array([0.0,1.0])

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               channels = 1):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from max of .25 to 1.
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        # images = np.multiply(images, 4)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(datafile,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=0,
                   fraction=1,
                   channels=1,
                   imagery='alff'):


    f = h5py.File(datafile,'r')
    sub = f["subjects"]
    ids = [x for x in sub]
    dims = (slice(5,55),slice(8,68),slice(4,52))


    filt_ids = []
    for ID in ids:
         if sub[ID].attrs.get('AGE_AT_SCAN') > 16.0 and sub[ID].attrs.get('SEX') == 1:
            filt_ids.append(ID)

    import random
    #random.seed(5)
    random.shuffle(filt_ids)

    train_ids = filt_ids[0:int(len(filt_ids)*.75)]
    test_ids = filt_ids[int(len(filt_ids)*.75):]

    train_ids = train_ids[0:int(len(train_ids)/fraction)]
    test_ids = test_ids[0:int(len(test_ids)/fraction)]

    chans = [x for x in sub[ids[0]]]
    chans = chans[:channels]

    dots = [np.zeros([50,60,48]) for x in range(channels)]
    # for c in range(channels):
    #     for x in range(42-c, 45-c):
    #         for y in range(2*c+13,2*c+16):
    #             for z in range(22,25):
    #                 dots[c][x,y,z] = (1. - .9) * np.random.random_sample() + 0.9

    def sub_imgs(ID):
        dot_img = sub[ID][chans[0]][dims]
        dot_img = dot_img / np.max(dot_img)
        # if int(sub[ID].attrs.get('DX_GROUP')) == 1:
        #     dot_img = dot_img + dots[0] * ( (0.4 - 0.2) * np.random.random_sample() + 0.2 )
        dot_img = np.amax(dot_img, axis=2)
        sub_img = np.ravel(dot_img)
        for i, chan in enumerate(chans[1:]):
            dot_img = sub[ID][chan][dims]
            dot_img = dot_img / np.max(dot_img)
            # if int(sub[ID].attrs.get('DX_GROUP')) == 1:
            #     dot_img = dot_img + dots[1+i] * ( (0.8 - 0.4) * np.random.random_sample() + 0.4 )
                # print("+")
            # else:
                # print("-")
            dot_img = np.amax(dot_img, axis=2)
            dot_img_r = np.ravel(dot_img)
            sub_img = np.vstack( (sub_img, dot_img_r ) )
        sub_img = sub_img.flatten(order='F')
        return sub_img


    train_imgs = sub_imgs(train_ids[0])
    test_imgs  = sub_imgs(test_ids[0])

    print("Loading Training Imagery...")
    for i, ID in enumerate(train_ids[1:]):
        if i%10 == 0:
            print('.',end="")
        train_imgs = np.vstack( (train_imgs, sub_imgs(ID)) )

    print("\nLoading Test Imagery...")
    for i, ID in enumerate(test_ids[1:]):
        if i%10 == 0:
            print('.',end="")
        test_imgs = np.vstack( (test_imgs, sub_imgs(ID)) )

    trainlabels = group_cat(sub[train_ids[0]].attrs.get('DX_GROUP'))
    for i,ID in enumerate(train_ids[1:]):
        trainlabels = np.vstack( (trainlabels, group_cat(sub[ID].attrs.get('DX_GROUP'))) )

    testlabels = group_cat(sub[test_ids[0]].attrs.get('DX_GROUP'))
    for i,ID in enumerate(test_ids[1:]):
        testlabels = np.vstack( (testlabels, group_cat(sub[ID].attrs.get('DX_GROUP'))) )

    validation = None

    train = DataSet(train_imgs, trainlabels, dtype=dtype, reshape=reshape)
    test = DataSet(test_imgs, testlabels, dtype=dtype, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)


def load_data(datafile='./data/AllSubjects4cat.hdf5'):
  return read_data_sets(datafile)
