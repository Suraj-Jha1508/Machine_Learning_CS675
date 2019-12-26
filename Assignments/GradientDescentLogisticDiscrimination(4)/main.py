import sys
from GradientDescentLogistic import GradientDescentLogistic

#Read Data File
print('Reading traindata file....')
DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]
for i in range(len(DataList)):
    DataList[i].append(1.0)

#Read Label File
Label={}
print('Reading trainlabel file...')
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       Label[int(a[1])] = int(a[0])
       x = f.readline()


stopping_condition=0.001
object=GradientDescentLogistic(DataList,Label,stopping_condition)

#Training by given Data set and their labels
object.training()

#Distance to the origin:
print('Distance from origin: {}'.format(object.distance_to_origin()))

#Predicting unlabeled Data set
predicted_label=object.prediction()
