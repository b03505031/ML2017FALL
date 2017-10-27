import xgboost as xgb
import sys
import numpy as np
import csv
x_train = './X_train'
y_train = './Y_train'
test = './X_test'
output = './xgb_output.csv'

#Reading train X
train_x=[]
rCounting = 0
with open(x_train,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader) #skip csv head
    for row in csvReader:
        train_x.append([1])
        for col in range(len(row)):
            train_x[len(train_x)-1].append(float(row[col]))

train_y=[]
with open(y_train,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader)
    for row in csvReader:
        train_y.append(float(row[0]))
       
test_x = []
with open(test,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader) #skip csv head
    for row in csvReader:
        test_x.append([1])
        for col in range(len(row)):
            test_x[len(test_x)-1].append(float(row[col]))
age = 3
flnwgt = 2
gain = 4
loss = 4
hours = 3
age_gain = 1


#Adding columns
for i in range(len(train_x)):
    for p in range(2,age+1):
        train_x[i].append(float(train_x[i][1]**p)) #age
    for p in range(2,flnwgt+1):
        train_x[i].append(float(train_x[i][2]**p)) #flnwgt
    for p in range(2,gain+1):
        train_x[i].append(float(train_x[i][4]**p)) #GAIN
    for p in range(2,loss+1):
        train_x[i].append(float(train_x[i][5]**p)) #loss
    for p in range(2,hours+1):
        train_x[i].append(float(train_x[i][6]**p)) #hours
    for p in range(1,age_gain+1):
        train_x[i].append(float((train_x[i][1]*train_x[i][4])**p)) #age*gain
for i in range(len(test_x)):
    for p in range(2,age+1):
        test_x[i].append(float(test_x[i][1]**p)) #age
    for p in range(2,flnwgt+1):
        test_x[i].append(float(test_x[i][2]**p)) #flnwgt
    for p in range(2,gain+1):
        test_x[i].append(float(test_x[i][4]**p)) #GAIN
    for p in range(2,loss+1):
        test_x[i].append(float(test_x[i][5]**p)) #loss
    for p in range(2,hours+1):
        test_x[i].append(float(test_x[i][6]**p)) #hours
    for p in range(1,age_gain+1):
        test_x[i].append(float((test_x[i][1]*test_x[i][4])**p)) #age*gain

x_t=np.array(train_x)
y_t=np.array(train_y)
x_testnp=np.array(test_x)
#x_t=x_t.T
#x_testnp=x_testnp.T
y_t=y_t.T
m = np.mean(x_t,axis=0)
std = np.std(x_t,axis=0)
x_t = x_t.T
for i in range(len(x_t)):
    if std[i] != 0:
        x_t[i] = (x_t[i]-m[i])/std[i]
x_t = x_t.T
m = np.mean(x_testnp,axis=0)
std = np.std(x_testnp,axis=0)
x_testnp = x_testnp.T
for i in range(len(x_testnp)):
    if std[i] != 0:
        x_testnp[i] = (x_testnp[i]-m[i])/std[i]
x_testnp = x_testnp.T
dtest=xgb.DMatrix(x_testnp)
dtrain=xgb.DMatrix(x_t,label=y_t)
param = {'max_depth': 6, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'error'
num_round = 100
evallist = [(dtrain, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist)
#bst=xgb.XGBClassifier(max_depth=6,objective='binary:logistic',nthread=4)
ypred=bst.predict(dtest)
print(ypred)
ofile = open(output, 'w')
ofile.write("id,label")
ofile.write('\n')
for i in range(len(ypred)):
	ofile.write(str(i+1) + ',')
	if ypred[i] >= 0.5:
		ofile.write('1')
	else:
		ofile.write('0')
	ofile.write('\n')
ofile.close()