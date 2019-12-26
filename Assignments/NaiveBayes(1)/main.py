import sys
from NaiveBayes import NaiveBayes

#Read Data File
print('Reading traindata file....')
DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]

#Read Label File
Label={}
print('Reading trainlabel file...')
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       Label[int(a[1])] = int(a[0])
       x = f.readline()

object=NaiveBayes(DataList,Label)

#training NaiveBayes Model
object.training()

#prediction
predicted_labels=object.prediction()