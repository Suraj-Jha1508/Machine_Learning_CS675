import sys
from bagging import Bagging

Label = {}
# Read Data file
print('Reading traindata file....')
DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]
# Reading Label file
print('Reading trainlabel file...')
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       Label[int(a[1])] = int(a[0])
       x = f.readline()

object=Bagging(DataList,Label)
result=object.boot_strap(100)
