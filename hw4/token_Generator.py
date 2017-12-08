import numpy as np
import keras
import pickle
import sys
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

trainFile='./training_label.txt'
semiFile='./training_nolabel.txt'
EMBEDDING_DIM=300
MAX_SEQUENCE_LENGTH=50

x_train=[]
#Establishing labeledtraining data
with open(trainFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
for i in range(len(lines)):
    x_train.append(((lines[i].split(' +++$+++ ')[1]).replace(" ' ","")).replace("\n",""))


with open(semiFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
for i in range(len(lines)):
    x_train.append((lines[i].replace(" ' ","")).replace("\n",""))


#tokening
tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_train)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)