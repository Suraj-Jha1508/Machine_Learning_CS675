import sys
from RandomHyperplane import RandomHyperplane



#Read Data File

DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]
#Read Label File
trainlabels={}
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       trainlabels[int(a[1])] = int(a[0])
       x = f.readline()

k=int(sys.argv[3])
object=RandomHyperplane(DataList,trainlabels,k)
b_hyp,minerror_hyp,p2=object.predict_random_hyperplane_data()


value=[]
for i in range(len(DataList)):
    if trainlabels.get(i) is None:
        value.append(i)

for i in range(len(value)):
    print(p2[i],value[i])




