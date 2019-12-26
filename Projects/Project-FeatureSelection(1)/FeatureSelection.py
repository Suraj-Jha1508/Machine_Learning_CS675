
class FeatureSelection():
    def __init__(self,data,label):
        '''
    
        Function: Constructor
        
        Input:    data,
                  labels
        Output:   NA
    
        '''
        self.DataList=data
        self.Label=label
        self.row=len(self.DataList)
        self.col=len(self.DataList[0])

    def mean(self,list1):
        '''
    
        Function: mean calculation of a list
        
        Input:    list
        Output:   mean of a list
    
        '''
        mean = 0
        for i in list1:
            mean += i
        return mean / len(list1)
    def f_scoreCal(self,col):
        '''
    
        Function: f-score calculator
        
        Input:    column number
        Output:   f-score of the column
    
        '''
        listcol = [item[col] for item in self.DataList]
        x_mean = self.mean(listcol)
        x0 = []
        x1 = []
        n0 = 0
        for i in range(self.row):
            # print('label:',label[i])
            if self.Label[i] == 0:
                # print(i)
                x0.append(listcol[i])
                n0 += 1
            else:
                x1.append(listcol[i])
        x0_mean = self.mean(x0)
        x1_mean = self.mean(x1)
        # print(x0_mean,x1_mean,x_mean)
        d0 = 0
        d1 = 0
        for i in x0:
            d0 += (i - x0_mean) ** 2
        for i in x1:
            d1 += (i - x1_mean) ** 2
        d0 /= (n0 - 1)
        d1 /= ((self.row - n0) - 1)
        if (d0 + d1) == 0:
            return -1
        return (((x0_mean - x_mean) ** 2 + (x1_mean - x_mean) ** 2) / (d0 + d1))

    def column_selection(self):
        '''
    
        Function: Feature Selection on the basis of f-score
        
        Input   : NA
        Return  : selected feature
    
        '''
        f_score = {}
        for j in range(self.col):
            f=self.f_scoreCal(j)
            f_score[f] = j
        vals = sorted(f_score.keys(), reverse=True)[:15]
        featured_col = []
        for i in vals:
            featured_col.append(f_score[i])
        print('Below are the feature selected as per f-score\n{}'.format(featured_col))
        return featured_col
