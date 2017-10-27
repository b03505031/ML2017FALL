import sys
import numpy as np
import csv
x_train =sys.argv[3] #'./X_train'
y_train = sys.argv[4]#'./Y_train'
test = sys.argv[5]#'./X_test'
output = sys.argv[6] #'./output.csv'
norm = 0
def sgm(z):
    return np.clip(1.0/(1.0+np.exp(-z)),0.00000000000001,0.99999999999999)

def generative(m1,m2,sigma,x,c1,c2):
    sigmaIn = np.linalg.inv(sigma)
    w=np.dot(np.transpose(m1-m2),sigmaIn)
    b=(-0.5)*np.dot(np.dot(np.transpose(m1),sigmaIn),m1) + (0.5)*np.dot(np.dot(np.transpose(m2),sigmaIn),m2) +np.log(float(c1)/c2)
    z = np.dot(w,x.T) + b
    return sgm(z)


#Reading train X
train_x=[]
rCounting = 0
with open(x_train,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader) #skip csv head
    for row in csvReader:
        train_x.append([])
        for col in range(len(row)):
            train_x[len(train_x)-1].append(float(row[col]))

#Read trai Y
train_y=[]
with open(y_train,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader)
    for row in csvReader:
        train_y.append(float(row[0]))

#Read test X
test_x=[]
with open(test,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader) #skip csv head
    for row in csvReader:
        test_x.append([])
        for col in range(len(row)):
            test_x[len(test_x)-1].append(float(row[col]))

#Numpy array
x_np = np.array(train_x)
y_np = np.array(train_y)
test_np = np.array(test_x)

#norm
if(norm==1):
    print("norming")
    m = np.mean(x_np,axis=1)
    std = np.std(x_np,axis=1)
    x_np = x_np.T
    for i in range(106):
        if std[i] != 0:
            x_np[i] = (x_np[i]-m[i])/std[i]
    x_np = x_np.T
    m2 = np.mean(x_np,axis=0)
    std2 = np.std(x_np,axis=0)
    test_np = test_np.T
    for i in range(106):
        if std2[i] != 0:
            test_np[i] = (test_np[i]-m2[i])/std2[i]
    test_np = test_np.T

featureNO = 106

#calculating Share Sigma
m1=np.zeros(featureNO)
m2=np.zeros(featureNO)
c1=0
c2=0
for i in range(len(train_y)):
    if y_np[i] ==1:
        m1+=x_np[i]
        c1+=1
    else:
        m2+=x_np[i]
        c2+=1
m1/=c1
m2/=c2

s1=np.zeros((featureNO,featureNO))
s2=np.zeros((featureNO,featureNO))
for i in range(len(x_np)):
    if y_np[i]==1:
        s1+=np.dot(np.transpose([x_np[i]-m1]),[x_np[i]-m1])
    else:
        s2+=np.dot(np.transpose([x_np[i]-m2]),[x_np[i]-m2])
sigma = (s1+s2)/float(len(x_np))

with open(output,'w') as file:
    file.write("id,label\n")
    ans = generative(m1,m2,sigma,test_np,c1,c2)
    for i in range(len(ans)):
        file.write(str(i+1)+',')
        if ans[i]>=0.5:
            file.write('1\n')
        else:
            file.write('0\n')
