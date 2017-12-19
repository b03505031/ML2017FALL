import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K

test = sys.argv[1]
out = sys.argv[2]
mean = 3.58171208604
std = 1.11689766115
def rmse(y_true,y_pred):
    return K.sqrt(K.mean((y_pred - y_true)**2))

test = pd.read_csv(test)
user = test['UserID'].values
movie = test['MovieID'].values

arguments = ['5', '7', '8' ,'9', '10', '11', '12', '12', '14']
count = len(arguments)
print("Total "+str(count)+" model to be predicted")

model = load_model('model_'+arguments[0]+'.h5', custom_objects={"rmse":rmse})
print("Predicting: "+'model_'+arguments[0]+'.h5')
prediction = model.predict([user, movie])

for i in range(1,count):
	print("Predicting: "+'model_'+arguments[i]+'.h5')
	model=load_model('model_'+arguments[i]+'.h5', custom_objects={"rmse":rmse})
	if int(arguments[i]) is not 12:
		prediction+=model.predict([user, movie])
	else:
		print("restore norming")
		prediction+=((model.predict([user, movie])*std)+mean)

prediction /= count

with open(out, 'w') as ans:
	print('TestDataID,Rating', file=ans)
	for i in range(len(prediction)):
		print(str(i+1)+","+str(prediction[i][0]),file=ans)