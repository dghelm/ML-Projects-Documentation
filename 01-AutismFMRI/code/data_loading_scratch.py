from __future__ import print_function
import numpy as np
import load_fmri_multich_minusmean as input_data

CHANNELS = 4

RETRAIN = True
MODEL_FILE_DIR = "./models/"
MODEL_FILE = "cae_mch_161114-004512_4999.ckpt"
LOAD_FEATURES = False
CLASSIFY = True
RESET_GLOBAL_STEP = False
NEW_GLOBAL_STEP = 0


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
