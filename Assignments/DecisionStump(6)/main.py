import sys
from Gini import Gini

# Reading Data File
print('Reading traindata file...')
DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]

# Reading Label file as dictionary
print('Reading trainlabel file...')
Label = {}
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       Label[int(a[1])] = int(a[0])
       x = f.readline()

object=Gini(DataList,Label)
Gini,Split,Col=object.column_selection()
print('Column:{} has minimum Gini value:{} with Slpit:{}'.format(Col,Gini,Split))
