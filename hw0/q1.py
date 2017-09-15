import sys
with open(sys.argv[1],'r') as myfile:
    data = myfile.read()
origList = data.split()
resultList=[]
num = 0
for i in origList:
    location=-1
    for j in resultList:
        if i == j :
            location = resultList.index(j)
    if location==-1 :
        resultList.append(i)
        resultList.append(num)
        resultList.append(1)
        num = num + 1
    else:
        resultList[location+2] = resultList[location+2]+1
nowLoc=0
ans = ''
for i in resultList:
    if  nowLoc % 3 == 0:
        ans+=(str(i))
        ans+=(' ')
        nowLoc = nowLoc+1;
    elif nowLoc % 3 == 1:
        ans+=(str(i))
        ans+=(' ')
        nowLoc+=1
    elif nowLoc != 3*num-1:
        ans+=(str(i))
        ans+=("\n")
        nowLoc+=1
    else:
        ans+=(str(i))
with open("Q1.txt","w") as output:
    output.write(ans)
