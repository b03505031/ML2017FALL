import numpy as np
import sys
import os
from sklearn.cluster import KMeans
import keras
from keras.layers import Dense,Input
from keras.models import Model,load_model



args=sys.argv[1:]
imageF=sys.argv[1]
tsneFprefix='./tsne_'
pcaFprefix='./pca_'
testCaseF=sys.argv[2]
resultF=sys.argv[3]
encodeModelF='./encode.h5'
DIM=64



def dnn_autoencode(data,dim):
    if os.path.exists(encodeModelF):
        print("Load "+encodeModelF+" from disk")
        encoder=load_model(encodeModelF)
        encoded_imgs = encoder.predict(data)
        return encoded_imgs
        
    else:
        print("Performing DNN Autoencoder, DIM="+str(dim))
        # this is our input placeholder
        input_img = Input(shape=(784,))
        # encoder layers
        encoded = Dense(512, activation='relu')(input_img)
        encoded = Dense(128, activation='relu')(encoded)
        encoder_output = Dense(dim)(encoded)

        # decoder layers
        decoded = Dense(128, activation='relu')(encoder_output)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(784, activation='tanh')(decoded)
        # construct the autoencoder model
        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoder_output)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(data, data,
                        epochs=20,
                        batch_size=256,
                        shuffle=True)
        encoder.save(encodeModelF)
        encoded_imgs = encoder.predict(data)
        return encoded_imgs

image=np.load(imageF)
image = image.astype('float32') / 255.
image.reshape(len(a),784)
data= dnn_autoencode(image,DIM)
res = KMeans(n_clusters=2, random_state=None,init='random', n_init=30,max_iter=500,algorithm='auto').fit_predict(data)
print(res)
ans=[]
with open(testCaseF,'r') as file:
    lines=file.readlines()
    for line in lines:
        if line[0] != 'I':
            line=line.split(",")
            a=int(line[1])
            b=int(line[2])
            if res[a]==res[b]:
                ans.append('1')
            else:
                ans.append('0')
with open(resultF,'w') as file:
        file.write("ID,Ans\n")
        for i in range(len(ans)):
            file.write(str(i))
            file.write(",")
            file.write(ans[i])
            file.write("\n")
