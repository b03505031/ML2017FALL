import keras
from keras.models import Sequential,load_model
from keras.layers import Activation,Dense,Dropout,MaxPooling2D,Flatten,Conv2D
from keras.optimizers import rmsprop,Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard
import keras.losses
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import sys
import csv
import pandas as pd
import numpy as np

import string
import random
import shutil
def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase+string.digits):
    return (''.join(random.choice(chars) for _ in range(size)))
rdnstr=id_generator()
#shutil.copy2('./cnn.py','./'+rdnstr+'.py')

#ARGS
output=sys.argv[2]#'./emRes.csv'

test=sys.argv[1]#'./test.csv'
test_raw=pd.read_csv(test)

num_of_test=(len(test_raw))
width=48
length=48
numOfClass=7

te_x=test_raw['feature'].str.split(' ')
te_x = te_x.tolist()
te_x_np=np.array(te_x)
te_x_np = te_x_np.astype('float64')
te_x_np=te_x_np.reshape(num_of_test,width,length,1)
te_x_np/=255


model1 = load_model('./single_model.h5')
result=model1.predict_classes(te_x_np)


ofilee = open(output, 'w')
ofilee.write("id,label")
ofilee.write('\n')
for i in range(len(result)):
	ofilee.write(str(i)+','+str(result[i]))
	ofilee.write('\n')
ofilee.close()