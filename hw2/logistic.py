import sys
import numpy as np
import csv
x_train =sys.argv[3] #'./X_train'
y_train = sys.argv[4]#'./Y_train'
test = sys.argv[5]#'./X_test'
output = sys.argv[6] #'./output.csv'

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

#PARAM Area
ite = 2000
lr = 0.5
lamda = 10
age = 4
flnwgt = 2
gain = 4
loss = 4
hours = 4
age_gain = 1
norm=1
w_a=0
g_h=0

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
        train_x[i].append(float((train_x[i][1]*train_x[i][6])**p)) #age*gain
#Read trai Y
train_y=[]
with open(y_train,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader)
    for row in csvReader:
        train_y.append(float(row[0]))

poly = len(train_x[0])

train_x_np = np.array(train_x)
train_y_np = np.array(train_y)
print(train_x_np.size)
if(norm==1):
    print("norming")
    #norm
    m = np.mean(train_x_np,axis=0)
    std = np.std(train_x_np,axis=0)
    train_x_np = train_x_np.T
    for i in range(poly):
        if std[i] != 0:
            train_x_np[i] = (train_x_np[i]-m[i])/std[i]
    train_x_np = train_x_np.T


if(w_a==1):
    w_np = np.ones(poly)
elif(w_a==0):
    w_np = np.ones(poly)
if(g_h==1):
    gradient_history = np.ones(poly)
elif(g_h==0):
    gradient_history = np.zeros(poly)

#train section
for i in range(ite):
    z = np.dot(train_x_np,np.transpose(w_np))
    sigmoid = np.clip(1.0/(1.0+np.exp(-z)),0.00000000000001,0.99999999999999)
    w_cp = np.copy(w_np)
    w_cp[0] = 0.0
    gradient = np.dot(train_x_np.T,(sigmoid-train_y_np))+lamda*w_np
    gradient_history+=(gradient**2) #+2*lamda*w_np-lamda*w_np[0])
    w_np-=lr*gradient/np.sqrt(gradient_history)
    if(i%100==0):
        correctCount = 0
        for j in range(len(sigmoid)):
            if sigmoid[j]>=0.5 and train_y_np[j]==1:
                correctCount+=1
            elif sigmoid[j]<0.5 and train_y_np[j]==0:
                correctCount+=1
        print("acc : "+str(correctCount/len(train_x)))

#Reading Testing data
test_x = []
with open(test,'r') as file:
    csvReader = csv.reader(file)
    next(csvReader) #skip csv head
    for row in csvReader:
        test_x.append([1])
        for col in range(len(row)):
            test_x[len(test_x)-1].append(float(row[col]))

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

test_x_np = np.array(test_x)

if(norm==1):
    print("norming")
    #Norm
    m = np.mean(test_x_np,axis=0)
    std = np.std(test_x_np,axis=0)
    test_x_np = test_x_np.T
    for i in range(poly):
        if std[i] != 0:
            test_x_np[i] = (test_x_np[i]-m[i])/std[i]
    test_x_np = test_x_np.T

#Predicting
ofile = open(output, 'w')
ofile.write("id,label")
ofile.write('\n')
a_z = np.dot(test_x_np, w_np)
a_sigmoid = np.clip(1.0/(1.0+np.exp(-a_z)),0.00000000000001,0.99999999999999)
for i in range(len(a_sigmoid)):
	ofile.write(str(i+1) + ',')
	if a_sigmoid[i] >= 0.5:
		ofile.write('1')
	else:
		ofile.write('0')
	ofile.write('\n') 
ofile.close()