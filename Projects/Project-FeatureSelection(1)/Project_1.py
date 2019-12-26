from sklearn import svm
import sys
import math
import random
from FeatureSelection import FeatureSelection


#Read Data File
print('Reading traindata file....')
DataList = open(sys.argv[1]).readlines()
DataList = [line.split() for line in DataList]
DataList = [list(map(float, line)) for line in DataList]
#Read Label File
print('Reading trainlabel file...')
Label={}
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       Label[int(a[1])] = int(a[0])
       x = f.readline()

def crossValidation(newdata,newlabel,val_data,val_label):
    
    clf = svm.LinearSVC()
    clf.fit(newdata, newlabel)
    prediction = clf.predict(val_data)
    err = 0
    for i in range(0, len(prediction), 1):
        if (prediction[i] != val_label[i]):
            err = err + 1
    err = err / len(val_label)
    print('Accuracy',(1-err))

random.seed()
rowIDs = []
for i in range(0, len(DataList), 1):
    rowIDs.append(i)
#### Making a random train/validation split of ratio 90:10
newtrain = []
newlabels = []
validation = []
validationlabels = []

random.shuffle(rowIDs) #randomly reorder the row numbers
#print(rowIDs)

for i in range(0, int(.9*len(rowIDs)), 1):
    newtrain.append(DataList[i])
    newlabels.append(Label[i])
for i in range(int(.9*len(rowIDs)), len(rowIDs), 1):
    validation.append(DataList[i])
    validationlabels.append(Label[i])

print('Working on Feature Selection as per f-score...')
Object=FeatureSelection(newtrain,newlabels)
feature_col=Object.column_selection()
#print(feature_col)
featured_data=[]
for list1 in newtrain:
    l=[]
    for i in feature_col:
        l.append(list1[i])
    featured_data.append(l)
val_data=[]
for list1 in validation:
    l=[]
    for i in feature_col:
        l.append(list1[i])
    val_data.append(l)

crossValidation(featured_data,newlabels,val_data,validationlabels)
print('Reading testdata file...')
testdata = open(sys.argv[3]).readlines()
testdata = [line.split() for line in testdata]
testdata = [list(map(float, line)) for line in testdata]
test=[]
for list1 in testdata:
    l=[]
    for i in feature_col:
        l.append(list1[i])
    test.append(l)
clf = svm.LinearSVC(C=0.01)
clf.fit(featured_data, newlabels)
prediction = clf.predict(test)
for i in range(len(prediction)):
    print(str(prediction[i])+' '+str(i))

