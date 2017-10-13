#temp, wind speed, rain, co, hc, nox, o3
import sys
import csv
import numpy as np
import pandas as pd

orig_train = './train_no7.csv'
orig_data=[]
tr_data=[]
te_data=[]
Sets=[]

for i in range(3):
    Sets.append([])
for i in range(18):
    tr_data.append([])
    te_data.append([])
for i in range(3):
    for j in range(18):
        Sets[i].append([])
# AMB_TEM=0, PCH4=1,CO=2,NMHC=3,NO=4,NO2=5,NOx=6,O3=7,PM10=8,PM2.5=9
# RAINFALL=10,RH=11,SO2=12,THC=13,WD_HR=14,WIND_DIREC=15,WIND_SPEED=16,WS_HR=17
with open(orig_train,'r',encoding = 'Big5') as file:
    for row in csv.reader(file):
        orig_data.append(row)


for i in range(1,3961):
    if orig_data[i][2] == "AMB_TEMP":
        for column in range(3, len(orig_data[i])):
            tr_data[0].append(float(orig_data[i][column]))

    if orig_data[i][2] == "CH4":
        for column in range(3, len(orig_data[i])):
            tr_data[1].append(float(orig_data[i][column]))
    if orig_data[i][2] == "CO":
        for column in range(3, len(orig_data[i])):
            tr_data[2].append(float(orig_data[i][column]))
    if orig_data[i][2] == "NMHC":
        for column in range(3, len(orig_data[i])):
            tr_data[3].append(float(orig_data[i][column]))
    if orig_data[i][2] == "NO":
        for column in range(3, len(orig_data[i])):
            tr_data[4].append(float(orig_data[i][column]))
    if orig_data[i][2] == "NO2":
        for column in range(3, len(orig_data[i])):
            tr_data[5].append(float(orig_data[i][column]))
    if orig_data[i][2] == "NOx":
        for column in range(3, len(orig_data[i])):
            tr_data[6].append(float(orig_data[i][column]))
    if orig_data[i][2] == "O3":
        for column in range(3, len(orig_data[i])):
            tr_data[7].append(float(orig_data[i][column]))
    if orig_data[i][2] == "PM10":
        for column in range(3, len(orig_data[i])):
            tr_data[8].append(float(orig_data[i][column]))
    if orig_data[i][2] == "PM2.5":
        for column in range(3, len(orig_data[i])):
            tr_data[9].append(float(orig_data[i][column]))
    if orig_data[i][2] == "RAINFALL":
        for column in range(3, len(orig_data[i])):
            if orig_data[i][column] == "NR":
                tr_data[10].append(0.0)
            else:
                tr_data[10].append(float(orig_data[i][column]))
    if orig_data[i][2] == "RH":
        for column in range(3, len(orig_data[i])):
            tr_data[11].append(float(orig_data[i][column]))
    if orig_data[i][2] == "SO2":
        for column in range(3, len(orig_data[i])):
            tr_data[12].append(float(orig_data[i][column]))
    if orig_data[i][2] == "THC":
        for column in range(3, len(orig_data[i])):
            tr_data[13].append(float(orig_data[i][column]))
    if orig_data[i][2] == "WD_HR":
        for column in range(3, len(orig_data[i])):
            #tr_data[14].append(float(orig_data[i][column]))
            dir = float(orig_data[i][column])

            if(dir>=135 and dir<315):
                tr_data[14].append(float(dir-135))
            else:
                if(dir<135):
                    tr_data[14].append(float(dir-135))
                else:
                    tr_data[14].append(float(dir-360-135))
            '''
            if(dir<=45):
                tr_data[14].append(float(dir+45))
            elif(dir>45 and dir<=225):
                tr_data[14].append(float(135-dir))
            else:
                tr_data[14].append(float(dir-135))
            '''
    if orig_data[i][2] == "WIND_DIREC":
        for column in range(3, len(orig_data[i])):
            tr_data[15].append(float(orig_data[i][column]))
    if orig_data[i][2] == "WIND_SPEED":
        for column in range(3, len(orig_data[i])):
            tr_data[16].append(float(orig_data[i][column]))
    if orig_data[i][2] == "WS_HR":
        for column in range(3, len(orig_data[i])):
            tr_data[17].append(float(orig_data[i][column]))
#x= 1*w + sum(all item of 9 hours after first hour * w)
#y= pm2.5 of 10th hour
# training part
AMB_TEMP =   2
CH4 =        7
CO =         7
NMHC =       0
NO =         7
NO2 =        7
NOx =        7
O3 =         7
PM10 =       7
PM10s =      7
PM25 =       9
PM25s =      9
RAINFALL =   2
RH =         2
SO2 =        7
THC =        0
WD_HR =      0
WIND_DIREC = 0
WIND_SPEED = 0
WS_HR =      0
WSWD =      1
lamda=0
ite=200000
startIndex=10
ifSave = 1
ifKill = 1
resFile = './best.csv'
poly = AMB_TEMP + CH4 + CO + NMHC + NO + NO2 + NOx + O3 + PM10 + PM10s + PM25 + PM25s + RAINFALL + RH + SO2 + THC + WD_HR + WIND_DIREC + WIND_SPEED + WS_HR + WSWD + 1
x=[]
y=[]
Xindex=0
for month in range(11):
    for hour in range(471):
        checkAva = 1
        for hour_after in range(PM25):
            if((float(tr_data[9][month * 480 + hour + hour_after + 9 - PM25])==-1)  or (float(tr_data[9][month*480+hour+9])==-1)):
                checkAva = 0
        if(checkAva==1 or ifKill==0):
            x.append([1])
            #amb_tem
            if(AMB_TEMP!=0):
                for hour_after in range(AMB_TEMP):
                    x[Xindex].append(tr_data[0][month*480+hour+hour_after+startIndex-AMB_TEMP])
            #pch4
            if(CH4!=0):
                for hour_after in range(CH4):
                    x[Xindex].append(tr_data[1][month*480+hour+hour_after+startIndex-CH4])
            #co
            if(CO!=0):
                for hour_after in range(CO):
                    x[Xindex].append(tr_data[2][month*480+hour+hour_after+startIndex-CO])
            #nmhc
            if(NMHC!=0):
                for hour_after in range(NMHC):
                    x[Xindex].append(tr_data[3][month*480+hour+hour_after+startIndex-NMHC])
            #no
            if(NO!=0):
                for hour_after in range(NO):
                    x[Xindex].append(tr_data[4][month*480+hour+hour_after+startIndex-NO])
            #no2
            if(NO2!=0):
                for hour_after in range(NO2):
                    x[Xindex].append(tr_data[5][month*480+hour+hour_after+startIndex-NO2])
            #nox
            if(NOx!=0):
                for hour_after in range(NOx):
                    x[Xindex].append(tr_data[6][month*480+hour+hour_after+startIndex-NOx])
            #o3
            if(O3!=0):
                for hour_after in range(O3):
                    x[Xindex].append(tr_data[7][month*480+hour+hour_after+startIndex-O3])
            #pm10
            if(PM10!=0):
                for hour_after in range(PM10):
                    x[Xindex].append(tr_data[8][month*480+hour+hour_after+startIndex-PM10])
            # pm10 squ
            if(PM10s!=0):
                for hour_after in range(PM10s):
                    x[Xindex].append(tr_data[8][month * 480 + hour + hour_after+startIndex-PM10s]**2)
            #pm2.5
            if(PM25!=0):
                for hour_after in range(PM25):
                    x[Xindex].append(tr_data[9][month*480+hour+hour_after+9-PM25])
            # pm2.5 squ
            if(PM25s!=0):
                for hour_after in range(PM25s):
                    x[Xindex].append(tr_data[9][month * 480 + hour + hour_after+9-PM25]**2)
            #rainfall
            if(RAINFALL!=0):
                for hour_after in range(RAINFALL):
                    x[Xindex].append(tr_data[10][month*480+hour+hour_after+startIndex-RAINFALL])
            #rh
            if(RH!=0):
                for hour_after in range(RH):
                    x[Xindex].append(tr_data[11][month*480+hour+hour_after+startIndex-RH])
            #so2
            if(SO2!=0):
                for hour_after in range(SO2):
                    x[Xindex].append(tr_data[12][month*480+hour+hour_after+startIndex-SO2])
            #thc
            if(THC!=0):
                for hour_after in range(THC):
                    x[Xindex].append(tr_data[13][month*480+hour+hour_after+startIndex-THC])

            #wd_hr
            if(WD_HR!=0):
                for hour_after in range(WD_HR):
                    x[Xindex].append(tr_data[14][month*480+hour+hour_after+startIndex-WD_HR])
            #wind_direction
            if(WIND_DIREC!=0):
                for hour_after in range(WIND_DIREC):
                    x[Xindex].append(tr_data[15][month*480+hour+hour_after+startIndex-WIND_DIREC])
            #wind_speed
            if(WIND_SPEED!=0):
                for hour_after in range(WIND_SPEED):
                    x[Xindex].append(tr_data[16][month*480+hour+hour_after+startIndex-WIND_SPEED])
            #ws_hr
            if(WS_HR!=0):
                for hour_after in range(WS_HR):
                    x[Xindex].append(tr_data[17][month*480+hour+hour_after+startIndex-WS_HR])
            # wswd
            if (WSWD != 0):
                for hour_after in range(WSWD):
                    x[Xindex].append(tr_data[17][month * 480 + hour + hour_after + startIndex - WSWD]*tr_data[14][month * 480 + hour + hour_after + startIndex - WSWD])
            Xindex+=1
            #pm2.5 result
            y.append(tr_data[9][month*480+hour+9])
x_array=np.array(x)
y_array=np.array(y)
w=np.zeros(poly)
gradient_history=np.zeros(poly)
learning_rate = 0.5
for i in range(ite):
    y=np.dot(x_array,w)
    gradient = 2*np.dot(x_array.T,(y-y_array))
    gradient_history+=(gradient**2+2*lamda*w.sum())
    w-= (learning_rate*gradient)/np.sqrt(gradient_history)
    if(i%1000==0):
        print(np.sqrt(np.mean(((y_array - y) ** 2))))

np.savetxt("weight,csv", w, delimiter=",")

#test.csv
file_test = open('./test.csv', 'r', encoding='Big5')

test_table = []
for row in csv.reader(file_test):
	tmp = []
	for column in range(2, 11):
		if row[column] != "NR":
			tmp.append(float(row[column]))
		else:
			tmp.append(0.0)
	test_table.append(tmp)
file_test.close()

for i in range(240):
    for j in range(9):
        if(float(test_table[18*i+8][j]==-1)):
            test_table[18 * i + 8][j] = test_table[18 * i + 8][j-1]
        if (float(test_table[18 * i + 9][j] == -1)):
            test_table[18 * i + 9][j] = test_table[18 * i + 9][j - 1]

test_x = []
for i in range(240):
    test_x.append([1])
    if(AMB_TEMP!=0):
        for j in range(9-AMB_TEMP, 9):
            test_x[i].append(test_table[18 * i + 0][j])
    if(CH4!=0):
        for j in range(9-CH4, 9):
            test_x[i].append(test_table[18 * i + 1][j])
    if(CO!=0):
        for j in range(9-CO, 9):
            test_x[i].append(test_table[18 * i + 2][j])
    if(NMHC!=0):
        for j in range(9-NMHC, 9):
            test_x[i].append(test_table[18 * i + 3][j])
    if(NO!=0):
        for j in range(9-NO, 9):
            test_x[i].append(test_table[18 * i + 4][j])
    if(NO2!=0):
        for j in range(9-NO2, 9):
            test_x[i].append(test_table[18 * i + 5][j])
    if(NOx!=0):
        for j in range(9-NOx, 9):
            test_x[i].append(test_table[18 * i + 6][j])
    if (O3 != 0):
        for j in range(9-O3, 9):
            test_x[i].append(test_table[18 * i + 7][j])
    if (PM10 != 0):
        for j in range(9-PM10, 9):
            test_x[i].append(test_table[18 * i + 8][j])
    if(PM10s != 0):
        for j in range(9-PM10s, 9):
            test_x[i].append(test_table[18 * i + 8][j]**2)
    if(PM25 != 0):
        for j in range(9-PM25, 9):
            test_x[i].append(test_table[18 * i + 9][j])
    if(PM25s != 0):
        for j in range(9-PM25s, 9):
            test_x[i].append(test_table[18 * i + 9][j]**2)
    if(RAINFALL != 0):
        for j in range(9-RAINFALL, 9):
            test_x[i].append(test_table[18 * i + 10][j])
    if(RH!=0):
        for j in range(9-RH, 9):
            test_x[i].append(test_table[18 * i + 11][j])
    if(SO2!=0):
        for j in range(9-SO2, 9):
            test_x[i].append(test_table[18 * i + 12][j])
    if(THC!=0):
        for j in range(9-THC, 9):
            test_x[i].append(test_table[18 * i + 13][j])
    if (WD_HR != 0):
        for j in range(9 - WD_HR, 9):
            test_x[i].append(test_table[18 * i + 14][j])
    if (WIND_DIREC != 0):
        for j in range(9 - WIND_DIREC, 9):
            test_x[i].append(test_table[18 * i + 15][j])
    if (WIND_SPEED != 0):
        for j in range(9 - WIND_SPEED, 9):
            test_x[i].append(test_table[18 * i + 16][j])
    if (WS_HR != 0):
        for j in range(9 - WS_HR, 9):
            test_x[i].append(test_table[18 * i + 17][j])
    if (WSWD != 0):
        for j in range(9 - WSWD, 9):
            test_x[i].append(test_table[18 * i + 17][j]*test_table[18 * i + 14][j])


testx = np.array(test_x)
testy = np.dot(testx,w)

if(ifSave==1):
    ofile = open(resFile, 'w')
    ofile.write("id,value")
    ofile.write('\n')
    for i in range(240):
        ofile.write("id_")
        ofile.write(str(i))
        ofile.write(",")
        ofile.write(str(testy[i]))
        ofile.write('\n')
    ofile.close()
