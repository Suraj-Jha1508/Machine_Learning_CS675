import random

class Gini():
    def gini_cal(self,col,data,label):
        '''
    
        Function: Calculating gini value and split for columns
        
        Input   : column,
                  data,
                  label
        Return  : gini value,
                  split
    
        '''
        rows=len(data)
        self.listcol = [item[col] for item in data]
        self.keys = sorted(range(len(self.listcol)), key=lambda k: self.listcol[k])
        self.listcol.sort()
        min_gini_val = float('inf')
        split=0
        for k in range(1,rows):
            lsize = k
            rsize = rows - k
            lp = 0
            rp = 0
            for l in range(rows):
                if (label[self.keys[l]] == 1) and l<k:
                    lp += 1
                if (label[self.keys[l]] == 1) and l>=k:
                    rp += 1
            gini = (lsize / rows) * (lp / lsize) * (1 - lp / lsize) + (rsize / rows) * (rp / rsize) * (1 - rp / rsize)
            if min_gini_val > gini:
                min_gini_val = gini
                split=k
        return min_gini_val,split

    def column_selection(self,data,label):
        '''
    
        Function: Selecting Column with least Gini
        
        Input   : data,
                  label
        Return  : column split,
                  column number
    
        '''
        overall_min_gini=float('inf')
        col_split=0
        col_num=0
        for j in range(len(data[0])):
            gini_val,split_val=self.gini_cal(j,data,label)
            if overall_min_gini>gini_val:
                overall_min_gini=gini_val
                col_split=split_val
                if col_split!=0:
                    col_split=(self.listcol[col_split] + self.listcol[col_split - 1]) / 2
                col_num=j
        print('column:{} split value :{}'.format(col_num,col_split))
        return col_split,col_num

class Bagging(Gini):
    def __init__(self,data,labels):
        '''
    
        Function: Constructor
        
        Input   : data,
                  labels
        Return  : NA
    
        '''
        self.data=data
        self.labels=labels
        self.rows=len(self.data)
        self.cols=len(self.data[0])
        self.predict_label={}
        self.labeled_key = []
        for i in range(self.rows):
            if (self.labels.get(i) == None):
                self.predict_label[i]=0
            else:
                self.labeled_key.append(i)

    def prediction(self,split,col):
        '''
    
        Function: Prediction based on majority of prediction done by n trained model
        
        Input   : split,
                  column
        Return  : NA
    
        '''
        count_0=0
        count_1=0
        flag=1
        for i in range(self.rows):
            if (self.labels.get(i) != None):
                if (self.data[i][col] < split):
                    if (self.labels.get(i) == 0):
                        count_0 += 1
                    if (self.labels.get(i) == 1):
                        count_1 += 1
        if (count_0 > count_1):
            flag = 0
        for i in range(self.rows):
            if (self.labels.get(i) == None):
                if (self.data[i][col] < split):
                    self.predict_label[i]=flag
                else:
                    self.predict_label[i] =1-flag
    def boot_strap(self,boot):
        '''
    
        Function: Random selection of trainng data to train the model
        
        Input   : boot
        Return  : predicted_label
    
        '''
        for k in range(boot):
            boot_data=[]
            boot_label={}
            for i in range(len(self.labeled_key)):
                x=random.choice(self.labeled_key)
                if(self.labels.get(x) != None):
                    boot_data.append(self.data[x])
                    boot_label[i]=self.labels[x]
            split,col=self.column_selection(boot_data, boot_label)
            self.prediction(split,col)
        for i,j in self.predict_label.items():
            print(j,i)
        return self.predict_label


