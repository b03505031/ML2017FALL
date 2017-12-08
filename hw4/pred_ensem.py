import sys
import numpy as np
import csv
import math

import time
tic = time.clock()

af='a.csv'
bf='b.csv'
cf='c.csv'
outputFile='ensem.csv'
MAX_SEQUENCE_LENGTH=50


res=[]
with open(af,'r',encoding='utf-8') as file:
    for row in csv.reader(file):
        res.append(float(row[0].replace("[ ","").replace("]","")))
with open(bf,'r',encoding='utf-8') as file:
    i=0
    for row in csv.reader(file):
        res[i]+=(float(row[0].replace("[ ","").replace("]","")))
        i+=1
with open(cf,'r',encoding='utf-8') as file:
    i=0
    for row in csv.reader(file):
        res[i]+=(float(row[0].replace("[ ","").replace("]","")))
        i+=1
print(res)

with open(outputFile,'w') as file:
    file.write('id,label\n')
    for i in range(len(res)):
        if(res[i] >= 1.5):
            file.write(str(i)+','+str('1'))
        else:
            file.write(str(i)+','+str('0'))
        file.write('\n')

toc = time.clock()
with open('./log.txt','w') as file:
    tim=time.strftime("%Y-%m-%d %H:%M --> ",time.localtime())
    file.write(tim+str(toc-tic))
