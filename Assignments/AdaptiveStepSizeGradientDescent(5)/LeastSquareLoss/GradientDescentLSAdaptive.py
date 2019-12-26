import math
import random

class LeastSquareLoss():
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
                    self.dellf[j] += (self.labels[i]-LeastSquareLoss.DotProduct(self.w, self.data[i])) * self.data[i][j]

    def eta_cal(self):
        '''
    
        Function: minimum eta calculation
        
        Input   : NA
        Return  : minimum eta
    
        '''
        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001,
                    .00000000001]
        min_eta = 0
        min = self.objective_cal() 
        for k in range(0, len(eta_list), 1):
            eta = eta_list[k]
            for j in range(self.col):
                self.w[j] += self.dellf[j] * eta
            error = self.objective_cal()
            if min > error:
                min = error
                min_eta = eta
            for j in range(self.col):
                self.w[j] -= self.dellf[j] * eta
#        print("minimum_eta: ", min_eta)
        return min_eta
        
    def weight_update(self):
        '''
    
        Function: Updating weight
        
        Input   : NA
        Return  : NA
    
        '''
        eta=self.eta_cal()
        for j in range(self.col):
            self.w[j] += self.dellf[j] *eta

    def objective_cal(self):
        '''
    
        Function: objective calculation ((y-wTx)**2)
        
        Input   : NA
        Return  : error
    
        '''
        self.error = 0

        for i in range(self.row):
            if self.labels.get(i) is not None:
                self.error += (self.labels[i]-LeastSquareLoss.DotProduct(self.w, self.data[i]))**2

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


class GradientDecentLS(LeastSquareLoss):
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
                dp = LeastSquareLoss.DotProduct(self.w, self.data[i])
                if dp > 0:
                    prediction[i]=1
                    print("1", i)
                else:
                    prediction[i]=0
                    print("0", i)
        return prediction


