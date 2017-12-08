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

testFile=sys.argv[1]
tokenFile='./tokenizer.pickle'
modelFile='./final.h5'
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
    file.write('id,label\n')
    for i in range(len(prediction)):
        if(prediction[i] >= 0.5):
            file.write(str(i)+','+str('1'))
        else:
            file.write(str(i)+','+str('0'))
        file.write('\n')
