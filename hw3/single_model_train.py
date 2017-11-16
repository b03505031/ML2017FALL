import keras
from keras.models import Sequential
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
train=sys.argv[1]#'./train.csv'
train_raw=pd.read_csv(train)

num_of_train=(len(train_raw))
width=48
length=48
batch=100
epoch=200
numOfClass=7

tr_y=train_raw['label']
tr_x=train_raw['feature'].str.split(' ')
tr_y_np=np.array(tr_y)
tr_x = tr_x.tolist()
tr_x_np=np.array(tr_x)
tr_y_np=np_utils.to_categorical(tr_y_np,numOfClass)
tr_x_np = tr_x_np.astype('float64')
tr_x_np=tr_x_np.reshape(num_of_train,width,length,1)
tr_x_np/=255

#training data preprocess
idg = ImageDataGenerator(
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=6,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0,
    zoom_range=0,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None)
idg.fit(tr_x_np)

#CNN model
model = Sequential()
'''
keras.layers.Conv2D(filters, 
kernel_size, strides=(1, 1), 
padding='valid', 
data_format=None, 
dilation_rate=(1, 1), 
activation=None, 
use_bias=True, 
kernel_initializer='glorot_uniform', 
bias_initializer='zeros', 
kernel_regularizer=None, 
bias_regularizer=None, 
activity_regularizer=None, 
kernel_constraint=None, 
bias_constraint=None)
'''
model.add(Conv2D(24,(3,3),input_shape=(length,width,1)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(numOfClass))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.summary()

es=EarlyStopping(monitor='val_acc', patience=7, verbose=0, mode='auto')
rlr=ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
tb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#model.fit(tr_x_np,tr_y_np,batch_size=batch,epochs=epoch)
model.fit_generator(idg.flow(tr_x_np, tr_y_np, batch_size=batch),steps_per_epoch=(num_of_train//batch), epochs=epoch,callbacks=[tb])
model.save('cnn_model.h5')