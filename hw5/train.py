import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input,Embedding,Flatten,Dense,Dot,Add,Concatenate,Dropout
from keras.optimizers import Nadam,Adam
from keras.callbacks import EarlyStopping,TensorBoard, ModelCheckpoint
from keras import backend as K


def rmse(yt,yp):
    return K.sqrt(K.mean((yp - yt)**2))

trainF='train.csv'
usersF='users.csv'
movieF='movies.csv'


batch_size=4500
epochs=1300
embedding_dim=200
norm=False
bias='True'
uvei='ones'
uvdp=0.5
ubei='zeros'
mvei='zeros'
mvdp=0.5
mbei='zeros'

train=pd.read_csv(trainF)
MAX_USER_ID=train['UserID'].max()+1
MAX_MOVI_ID=train['MovieID'].max()+1
train=train.sample(frac=1)
user=(train['UserID'].values)
movi=(train['MovieID'].values)
rating=(train['Rating'].values)

if norm is True:
	print("Norming..")
	print("MEAN: "+str(np.mean(rating)))
	print("STD : "+str(np.std(rating)))
	rating=(rating-np.mean(rating))/np.std(rating)
	

#USER VECTOR AND ITS BIAS
ui = Input(shape=[1])
uv=Embedding(MAX_USER_ID,embedding_dim,embeddings_initializer=uvei)(ui)
uv=Flatten()(uv)
uv=Dropout(uvdp)(uv)
ub=Embedding(MAX_USER_ID,1,embeddings_initializer=ubei,trainable=bias)(ui)
ub=Flatten()(ub)

#MOVI VECTOR AND ITS BIAS
mi = Input(shape=[1])
mv=Embedding(MAX_MOVI_ID,embedding_dim,embeddings_initializer=mvei)(mi)
mv=Flatten()(mv)
mv=Dropout(mvdp)(mv)
mb=Embedding(MAX_MOVI_ID,1,embeddings_initializer=mbei,trainable=bias)(mi)
mb=Flatten()(mb)

res=Add()([Dot(axes=1)([uv,mv]),ub,mb])

model=Model(inputs=[ui,mi],outputs=res)
model.summary()

model.compile(loss=rmse,optimizer='Nadam',metrics=[rmse])

earlystopping = EarlyStopping(monitor='val_rmse', patience = 15, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='model_cp.h5',
							 verbose=1,
							 save_best_only=True,
							 monitor='val_rmse',
							 mode='min')
tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit([user,movi],rating,epochs=epochs,batch_size=batch_size,validation_split=0.1,callbacks=[earlystopping,checkpoint,tb])