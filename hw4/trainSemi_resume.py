#coding=utf-8
#argv1: cont, argv2:vectorfile, argv3:tokenfile
import numpy as np
import keras
import pickle
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Embedding,Dense,Dropout,Activation,GRU,LSTM
from keras.optimizers import Adam, RMSprop
import keras.losses
from keras.callbacks import EarlyStopping,TensorBoard, ModelCheckpoint
from keras import backend as K

cont = sys.argv[1]

trainFile='./training_label.txt'
semiFile='./training_nolabel.txt'
vectorFile='./vectors_gensim.txt'
tokenfile='./tokenizer.pickle'
loading_model='./model_cp.h5'

EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH=50
batch_size=700
epochs=50

#Establishing labeled training data
with open(trainFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
x_train=[]
y_train=[]
text=[]
for i in range(len(lines)):
    x_train.append(((lines[i].split(' +++$+++ ')[1]).replace(" ' ","")).replace("\n",""))
    y_train.append(lines[i].split(' +++$+++ ')[0])

#Establishing nolabel data
with open(semiFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
x_semi=[]
for i in range(len(lines)):
    x_semi.append((lines[i].replace(" ' ","")).replace("\n",""))

#tokening
with open(tokenfile, 'rb') as handle:
    tokenizer = pickle.load(handle)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(x_train)
x_train=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
y_train=np.array(y_train)
#establishing x_semi seq
semiSeq=tokenizer.texts_to_sequences(x_semi)
x_semi=pad_sequences(semiSeq,maxlen=MAX_SEQUENCE_LENGTH)

#establishing glove dictionary
embeddings_index={}
with open(vectorFile,'r',encoding='utf-8') as file:
    for line in file:
        val=line.split(" ")
        word=val[0]
        coefs=np.array(val[1:],dtype='float32')
        embeddings_index[word]=coefs

#establishing embedding matrix
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

#training with semi supervised
for semiloop in range(int(cont),8):
    #loading up model
    latestmodel=load_model('model_cp_'+str(semiloop-1)+'.h5')
    model=load_model('Initial_model.h5')
    print("\n##########################################################")
    print("Training Loop: "+str(semiloop))
    print("____________________________________________________________")
    print("Predicting unlabel data")
    if x_semi.shape[0] is not 0:
        semiPred= latestmodel.predict(x_semi)
        removelist=[]
        y_semi=[]
        x_toadd=[]
        for i in range(len(semiPred)):
            if semiPred[i]>=0.97:
                y_semi.append(1)
                removelist.append(i)
                x_toadd.append(x_semi[i])
            elif semiPred[i]<=0.03:
                y_semi.append(0)
                removelist.append(i)
                x_toadd.append(x_semi[i])
        y_semi=np.array(y_semi)
        y_train=np.append(y_train,y_semi)
        x_train=np.concatenate((x_train,np.array(x_toadd)),axis=0)
        x_semi=np.delete(x_semi,np.array(removelist),0)
    else:
        print("No unlabel data remained, end training")
        break
    print("____________________________________________________________")
    print("\n"+str(len(removelist))+" items added to training set")

    print("Now size:")
    print("   X_train: "+str(x_train.shape[0]))
    print("   Y_train: "+str(y_train.shape[0]))
    print("   X_nolab: "+str(x_semi.shape[0])+"\n")
    earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=('model_cp_'+str(semiloop)+'.h5'),
							 verbose=1,
							 save_best_only=True,
							 monitor='val_acc',
							 mode='max')
    tb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train,
            validation_data=(x_val,y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tb,checkpoint])
    print("##########################################################\n")
model.save('model_final.h5')

