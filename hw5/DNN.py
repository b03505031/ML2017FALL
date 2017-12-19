import numpy as np
import pandas as pd
import csv
from keras.layers import Embedding, Dropout, Dense, Input, Flatten, Concatenate
from keras.models import load_model, Model
from keras.optimizers import Adam, Adamax,Nadam,RMSprop
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint,TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

trainF='train.csv'
usersF='users.csv'
movieF='movies.csv'

epochs = 1000
batch_size = 4500
embedding_dim = 75

def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

#train data
train=pd.read_csv(trainF)
MAX_USER_ID=train['UserID'].max()+1
MAX_MOVI_ID=train['MovieID'].max()+1
train=train.sample(frac=1)
user=(train['UserID'].values)
movi=(train['MovieID'].values)
rating=(train['Rating'].values)

#user detail
with open(usersF,'r') as file:
	next(file)
	age_d={}
	sex_d={}
	occ_d={}
	for line in file.readlines():
		line=line.split("::")
		if line[1]=='M':
			sex_d[line[0]]=0
		else:
			sex_d[line[0]]=1	
		age_d[line[0]]=line[2]
		occ_d[line[0]]=line[3]

msex=[]
fsex=[]
age=[]
occ=[]
sex=[]
for i in range(len(user)):
	if int(sex_d[str(user[i])]) is 0:
		msex.append([1])
		fsex.append([0])
		sex.append([0])
	else:
		msex.append([0])
		fsex.append([1])
		sex.append([1])
	age.append([int(age_d[str(user[i])])])
	occ.append([int(occ_d[str(user[i])])])
sex=pad_sequences(sex)
msex=pad_sequences(msex)
fsex=pad_sequences(fsex)
age=pad_sequences(age)
occ=pad_sequences(occ)
MAX_AGE=age.max()+1
MAX_OCC=occ.max()+1


ui = Input(shape=[1])
uv=Embedding(MAX_USER_ID,embedding_dim,embeddings_initializer='ones')(ui)
uv=Flatten()(uv)

mi = Input(shape=[1])
mv=Embedding(MAX_MOVI_ID,embedding_dim,embeddings_initializer='zeros')(mi)
mv=Flatten()(mv)

msi = Input(shape=[1])
msv=Embedding(2,embedding_dim,embeddings_initializer='zeros')(msi)
msv=Flatten()(msv)

fsi = Input(shape=[1])
fsv=Embedding(2,embedding_dim,embeddings_initializer='zeros')(fsi)
fsv=Flatten()(fsv)

ai = Input(shape=[1])
av=Embedding(MAX_AGE,embedding_dim,embeddings_initializer='zeros')(ai)
av=Flatten()(av)

oi = Input(shape=[1])
ov=Embedding(MAX_MOVI_ID,embedding_dim,embeddings_initializer='zeros')(oi)
ov=Flatten()(ov)

fv = Concatenate()([uv,mv,msv,fsv,av,ov])
#dnn = Dense(512, activation='elu')(fv)
#dnn = Dropout(0.5)(dnn)
dnn = Dense(256, activation='elu')(fv)
dnn = Dropout(0.4)(dnn)
dnn = Dense(64, activation='elu')(dnn)
dnn = Dropout(0.3)(dnn)
dnn = Dense(1, activation='linear')(dnn)

model = Model(inputs=[ui,mi,msi,fsi,ai,oi], outputs=dnn)
model.summary()

model.compile(loss='mse',optimizer='Nadam',metrics=[rmse])

earlystopping = EarlyStopping(monitor='val_rmse', patience = 10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='model_cp.h5',
							 verbose=1,
							 save_best_only=True,
							 monitor='val_rmse',
							 mode='min')
tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit([user,movi,msex,fsex,age,occ], rating, 
	  epochs=epochs, 
	  batch_size=batch_size,
	  validation_split=0.2,
	  callbacks=[earlystopping,checkpoint,tb])