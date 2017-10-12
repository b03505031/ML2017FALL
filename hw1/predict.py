#temp, wind speed, rain, co, hc, nox, o3
import sys
import csv
import numpy as np
import pandas as pd


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
predictFile = sys.argv[1] #'./test.csv'
resFile = sys.argv[2] #'./best.csv'
poly = AMB_TEMP + CH4 + CO + NMHC + NO + NO2 + NOx + O3 + PM10 + PM10s + PM25 + PM25s + RAINFALL + RH + SO2 + THC + WD_HR + WIND_DIREC + WIND_SPEED + WS_HR + WSWD + 1

weight=[]
#load weight
with open('./weight.csv','r',encoding = 'Big5') as file:
    for row in csv.reader(file):
        weight.append(np.float64(row[0]))
w=np.array(weight)


#Readin predict
file_test = open(predictFile, 'r', encoding='Big5')
test_table = []
totalPred = 0
for row in csv.reader(file_test):
    totalPred+=1
    tmp = []
    for column in range(2, 11):
        if row[column] != "NR":
            tmp.append(float(row[column]))
        else:
            tmp.append(0.0)
    test_table.append(tmp)
file_test.close()
for i in range(int(totalPred/18)):
    for j in range(9):
        if(float(test_table[18*i+8][j]==-1)):
            test_table[18 * i + 8][j] = test_table[18 * i + 8][j-1]
        if (float(test_table[18 * i + 9][j] == -1)):
            test_table[18 * i + 9][j] = test_table[18 * i + 9][j - 1]
test_x = []
for i in range(int(totalPred/18)):
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
    for i in range(int(totalPred/18)):
        ofile.write("id_")
        ofile.write(str(i))
        ofile.write(",")
        ofile.write(str(testy[i]))
        ofile.write('\n')
    ofile.close()
