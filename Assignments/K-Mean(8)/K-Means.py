import sys
import random
import math

def dist(list1,list2):
    sum=0
    for i in range(len(list1)):
        sum+=(list1[i]-list2[i])**2
    return sum
    
def mean(list1):
    mean=[]
    for col in range(len(list1[0])):
        mean.append(round((sum([item[col] for item in list1]))/len(list1),2))
    return mean

#Read Data File
DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]
row=len(DataList)
col=len(DataList[0])

k_value=int(sys.argv[2])
Data=[]
mean_cluster=[]
split=row//k_value

data=DataList[:]
random.shuffle(DataList)

# Initial Clusters
for i in range(k_value):
    Cluster_Data=[]
    if i==k_value-1:
        x=row
    else:
        x=(i+1)*split
    for k in range(i*split,x):
        Cluster_Data.append(DataList[k])
    Data.append(Cluster_Data)
    mean_cluster.append(mean(Cluster_Data))


# K-Mean
prev_obj=100000
count=0
while True:
    count+=1
    obj=0
    for d in Data:
        j=d[:]
        for k in range(len(j)):
            min_dist=float('inf')
            for i in range(k_value):
                x=dist(mean_cluster[i],j[k])
                obj+=x
                if x<min_dist:
                    min_dist=x
                    idx=i
            Data[idx].append(j[k])
            d.remove(j[k])

    if prev_obj - obj == 0:
        break
    prev_obj = obj
    for i in range(k_value):
        if Data[i]==[]:
            mean_cluster[i]=[]
            for k in range(col):
                mean_cluster[i].append(0)
        else:
            mean_cluster[i]=mean(Data[i])

#printing datapoints and cluster
for i in range(row):
    for j in range(len(Data)):
        if data[i] in Data[j]:
            print(j,i)














