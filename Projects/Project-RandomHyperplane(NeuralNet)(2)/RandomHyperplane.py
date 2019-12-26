import random
from sklearn import svm

class RandomHyperplane():
    def __init__(self,data,label,k):
        self.data=data
        self.n=len(data)
        self.label=label
        self.only_label=[]
        for i in sorted(self.label):
            self.only_label.append(self.label[i])
        self.train_data=[]
        self.test_data=[]
        for i in range(self.n):
            self.data[i].append(1.0)
            if self.label.get(i) is not None:
                self.train_data.append(self.data[i])
            else:
                self.test_data.append(self.data[i])
        self.m=len(data[0])
        self.k=k

    @staticmethod
    def Sign(a):
        if a >= 0:
            return 1
        else:
            return -1

    @staticmethod
    def DotProduct(a, b):
        dp = map(lambda x, y: x * y, a, b)
        return sum(dp)

    def hyperplane_data(self):
        Z = []
        Z_prime = []
        for i in range(self.k):

            self.w = []
            for col in range(self.m-1):
                self.w.append(random.uniform(-1, 1))
            
            temp=[]
            for x in range(self.n-1):
                temp.append(RandomHyperplane.DotProduct(self.data[x],self.w))
           

            w0 = random.uniform(min(temp),max(temp))
            self.w.append(w0)

            if Z==[]:
                for row in range(len(self.train_data)):
                    Z.append([(1 + RandomHyperplane.Sign(RandomHyperplane.DotProduct(self.train_data[row], self.w))) / 2])
            else:
                for row in range(len(self.train_data)):
                    Z[row].append((1+RandomHyperplane.Sign(RandomHyperplane.DotProduct(self.train_data[row],self.w)))/2)

            if Z_prime==[]:
                for row in range(len(self.test_data)):
                    Z_prime.append([(1 + RandomHyperplane.Sign(RandomHyperplane.DotProduct(self.test_data[row], self.w))) / 2])
            else:
                for row in range(len(self.test_data)):
                    Z_prime[row].append((1+RandomHyperplane.Sign(RandomHyperplane.DotProduct(self.test_data[row],self.w)))/2)

        return Z,Z_prime

    def best_C(self,train,labels):
        random.seed()
        allCs = [.001, .01, .1, 1, 10, 100]
        error = {}
        for j in range(0, len(allCs), 1):
            error[allCs[j]] = 0
        rowIDs = []
        for i in range(0, len(train), 1):
            rowIDs.append(i)
        nsplits = 10
        for x in range(0, nsplits, 1):

            newtrain = []
            newlabels = []
            validation = []
            validationlabels = []

            random.shuffle(rowIDs)  # randomly reorder the row numbers
            # print(rowIDs)

            for i in range(0, int(.9 * len(rowIDs)), 1):
                    newtrain.append(train[i])
                    newlabels.append(labels[i])
            for i in range(int(.9 * len(rowIDs)), len(rowIDs),1):
                    validation.append(train[i])
                    validationlabels.append(labels[i])

            #### Predict with SVM linear kernel for values of C={.001, .01, .1, 1, 10, 100} ###
            for j in range(0, len(allCs), 1):
                C = allCs[j]
                clf = svm.LinearSVC(C=C)
                clf.fit(newtrain, newlabels)
                prediction = clf.predict(validation)

                err = 0
                for i in range(0, len(prediction), 1):
                    if (prediction[i] != validationlabels[i]):
                        err = err + 1

                err = err / len(validationlabels)
                error[C] += err

        bestC = 0
        minerror = 100
        keys = list(error.keys())
        for i in range(0, len(keys), 1):
            key = keys[i]
            error[key] = error[key] / nsplits
            if (error[key] < minerror):
                minerror = error[key]
                bestC = key

        return bestC, minerror
    def prediction_orignal(self):
        bestC,minerror=self.best_C(self.train_data,self.only_label)
        clf = svm.LinearSVC(C=100)
        clf.fit(self.train_data, self.only_label)
        prediction = clf.predict(self.test_data)
        return bestC,minerror,prediction

    def predict_random_hyperplane_data(self):
        train,test=self.hyperplane_data()
        bestC, minerror = self.best_C(train, self.only_label)
        clf = svm.LinearSVC(C=0.001)
        clf.fit(train, self.only_label)
        prediction = clf.predict(test)
        return bestC,minerror,prediction





