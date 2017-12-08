#coding=utf-8

import numpy as np
import keras
import pickle
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Embedding,Dense,Dropout,Activation,GRU,LSTM,Bidirectional,Conv1D,Flatten,Permute,MaxPooling1D
from keras.optimizers import Adam, RMSprop
import keras.losses
from keras.callbacks import EarlyStopping,TensorBoard, ModelCheckpoint
from keras import backend as K


trainFile=sys.argv[1]
vectorFile='./vectors_gensim.txt'
semiFile=sys.argv[2]
tokenfile='./tokenizer.pickle'

EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH=50
batch_size=500
epochs=70

#read training file
with open(trainFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
x_train=[]
y_train=[]
for i in range(len(lines)):
    x_train.append(((lines[i].split(' +++$+++ ')[1]).replace(" ' ","")).replace("\n",""))
    y_train.append(lines[i].split(' +++$+++ ')[0])

#token and sequence
with open(tokenfile, 'rb') as handle:
    tokenizer = pickle.load(handle)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(x_train)
x_train=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
y_train=np.array(y_train)

#loading glove dictionary
embeddings_index={}
with open(vectorFile,'r',encoding='utf-8') as file:
    for line in file:
        val=line.split(" ")
        word=val[0]
        coefs=np.array(val[1:],dtype='float32')
        embeddings_index[word]=coefs

#building weight matrix
embedding_matrix=np.zeros((len(word_index)+1,EMBEDDING_DIM))
for word,i in word_index.items():
    if i < len(word_index)+1:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

#validation set
indices = np.arange(x_train.shape[0])  
np.random.shuffle(indices) 
X = x_train[indices]
Y = y_train[indices]    
nvs = int(0.15 * X.shape[0] )
x_train = X[nvs:]
y_train = Y[nvs:]
x_val = X[:nvs]
y_val = Y[:nvs]


model = Sequential()
model.add(Embedding(len(word_index)+1,
                          EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=False))
model.add(Conv1D(150,kernel_size=3,activation='elu'))
model.add(Dropout(0.5))
model.add((LSTM(48, activation='tanh')))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
			  optimizer='rmsprop',
			  metrics=['acc'])
earlystopping = EarlyStopping(monitor='val_acc', patience = 13, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='final.h5',
							 verbose=1,
							 save_best_only=True,
							 monitor='val_acc',
							 mode='max')
tb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model.save('Initial_model.h5')
model.fit(x_train, y_train,
		  validation_data=(x_val,y_val),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[tb,earlystopping,checkpoint])
model.save('model.h5')

