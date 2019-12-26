import math
import random

class HingeLoss():
    def __init__(self,traindata,trainlabel,stop_cond):
        '''
    
        Function: Constructor
        
        Input   : TrainData,
                  TrainLabel,
                  stopping condition
        Return  : NA
    
        '''
        self.data=traindata
        self.labels=trainlabel
        self.stop_cond=stop_cond
        self.row=len(self.data)
        self.col=len(self.data[0])
        self.w = []
        for j in range(self.col):
            self.w.append(random.uniform(-0.01, 0.01))

    
    def gradient_cal(self):
        '''
    
        Function: gradient calculation ((y-wTx)*Xi)
        
        Input   : TrainData,
                  TrainLabels
        Return  : NA
    
        '''
        self.dellf = []
        for j in range(self.col):
            self.dellf.append(0)
        for i in range(self.row):
            if self.labels.get(i) is not None:
                for j in range(self.col):
                    if (HingeLoss.DotProduct(self.w, self.data[i]) * self.labels[i] < 1):
                        self.dellf[j] += - (self.labels[i] * self.data[i][j])


    def weight_update(self):
        '''
    
        Function: Updating weight
        
        Input   : NA
        Return  : NA
    
        '''
        eta=0.0001
        for j in range(self.col):
            self.w[j] -= self.dellf[j] *eta

    def objective_cal(self):
        '''
    
        Function: objective calculation ((y-wTx)**2)
        
        Input   : NA
        Return  : error
    
        '''
        self.error = 0

        for i in range(self.row):
            if self.labels.get(i) is not None:
                self.error += max(0, 1 - (self.labels[i] * HingeLoss.DotProduct(self.w, self.data[i])))

        return self.error

    @staticmethod
    def DotProduct(a,b):
        '''
    
        Function: Dot Product Calculation of two vector
        
        Input   : vector a,
                  vector b
        Return  : dot product of a and b
    
        '''
        dp = map(lambda x, y: x * y, a, b)
        return sum(dp)


class GradientDecentHL(HingeLoss):
    def training(self):
        '''
    
        Function: training model with labeled datapoint
        
        Input   : TrainData,
                  TrainLabels
        Return  : NA
    
        '''
        prev_error = float('inf')
        while True:
            self.gradient_cal()
            self.weight_update()
            self.objective_cal()
            if (prev_error - self.error) < self.stop_cond:
                break
            prev_error = self.error

    def distance_to_origin(self):
        '''
    
        Function: Hyperplane distance from origin
        
        Input   : NA
        Return  : distance
    
        '''
        dist = 0
        for j in range(self.col - 1):
            dist += (self.w[j] ** 2)
        dist = math.sqrt(dist)

        return abs((self.w[self.col - 1]) / dist)

    def prediction(self):
        '''
    
        Function: prediction of non label datapoints
        
        Input   : NA
        Return  : predicted label
    
        '''
        prediction={}
        for i in range(self.row):
            if self.labels.get(i) == None:
                dp = HingeLoss.DotProduct(self.w, self.data[i])
                if dp > 0:
                    prediction[i]=1
                    print("1", i)
                else:
                    prediction[i]=0
                    print("0", i)
        return prediction


