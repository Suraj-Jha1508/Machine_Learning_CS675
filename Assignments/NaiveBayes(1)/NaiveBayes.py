class NaiveBayes():
    def __init__(self,traindata,trainlabel):
        '''
    
        Function: Constructor
        
        Input   : TrainData,
                  TrainLabels
        Return  : NA
    
        '''
        self.traindata=traindata
        self.trainlabel=trainlabel
        self.row=len(self.traindata)
        self.col=len(self.traindata[0])
        self.mean=[[],[]]
        self.variance=[[],[]]
        for i in range(self.col):
            self.mean[0].append(0.01)
            self.mean[1].append(0.01)
            self.variance[0].append(0)
            self.variance[1].append(0)

    def training(self):
        '''
    
        Function: training the NaiveBayes Model with training datapoints
        
        Input   : NA
        Output  : NA      
    
        '''
        self.count=[0,0]
        for i in range(0, self.row, 1):
            if self.trainlabel.get(i) is not None and self.trainlabel.get(i) == 0:
                self.count[0]+=1
                for j in range(0, self.col, 1):
                    self.mean[0][j] += self.traindata[i][j]
            if self.trainlabel.get(i) is not None and self.trainlabel.get(i) == 1:
                self.count[1] += 1
                for j in range(0, self.col, 1):
                    self.mean[1][j] += self.traindata[i][j]

        for j in range(0, self.col, 1):
            self.mean[0][j] = self.mean[0][j] / self.count[0]
            self.mean[1][j] = self.mean[1][j] / self.count[1]

        for i in range(0, self.row, 1):
            if self.trainlabel.get(i) is not None and self.trainlabel.get(i) == 0:
                for j in range(0, self.col, 1):
                    self.variance[0][j] += ((self.traindata[i][j] - self.mean[0][j]) ** 2)
            if self.trainlabel.get(i) is not None and self.trainlabel.get(i) == 1:
                for j in range(0, self.col, 1):
                    self.variance[1][j] += ((self.traindata[i][j] - self.mean[1][j]) ** 2)

        for j in range(0, self.col, 1):
            self.variance[0][j] = self.variance[0][j] / self.count[0]
            self.variance[1][j] = self.variance[1][j] / self.count[1]

    def prediction(self):
        '''
    
        Function: Predicting the non label datapoints
        
        Input   : NA
        Return  : Predcted label dictionary       
    
        '''
        prediction={}
        for i in range(self.row):
            if self.trainlabel.get(i) is None:
                EucDist = [0,0]
                for j in range(self.col):
                    EucDist[0] += ((self.traindata[i][j] - self.mean[0][j]) ** 2 / self.variance[0][j])
                    EucDist[1] += ((self.traindata[i][j] - self.mean[1][j]) ** 2 / self.variance[1][j])
                if EucDist[0] < EucDist[1]:
                    prediction[i]=0
                    print("0", i)
                else:
                    prediction[i]=1
                    print("1", i)
        
        return prediction