import sys
import numpy as np

import math
import pickle
import keras
from keras.models import Sequential,load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,Dense,Dropout,Activation,GRU,LSTM
from keras.optimizers import Adam, RMSprop
import keras.losses
from keras.callbacks import EarlyStopping,TensorBoard
from keras import backend as K
import time
tic = time.clock()


testFile='./testing_data.txt'
tokenFile='./tokenizer.pickle'
modelFile=sys.argv[1]
outputFile=sys.argv[2]
MAX_SEQUENCE_LENGTH=50


i=0
te_x=[]
with open(testFile,'r',encoding='utf-8') as file:
     lines=file.readlines()
for i in range(len(lines)):
    if i is not 0:
       te_x.append(lines[i].replace(str(i-1)+',',"").replace(" ' ","").replace("\n",""))

with open(tokenFile, 'rb') as handle:
    tokenizer = pickle.load(handle)

#tokenizer=Tokenizer()

#tokenizer.fit_on_texts(te_x)
sequences=tokenizer.texts_to_sequences(te_x)
x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
model=load_model(modelFile)
prediction = model.predict(x_test)

with open(outputFile,'w') as file:
    for i in range(len(prediction)):
        file.write(str(prediction[i])+'\n')

toc = time.clock()
with open('./log.txt','w') as file:
    tim=time.strftime("%Y-%m-%d %H:%M --> ",time.localtime())
    file.write(tim+str(toc-tic))
