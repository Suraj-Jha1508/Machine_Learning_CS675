class Gini():
    def __init__(self,data,labels):
        '''
    
        Function: Constructor
        
        Input:    TrainData,
                  TrainLabels
    
        '''
        self.data=data
        self.labels=labels
        self.rows=len(self.data)
        self.cols=len(self.data[0])

    def gini_cal(self,col):
    
        '''
    
        Function: Calculating gini value and split for columns
        
        Input:    Column number
        Output:   minimum Gini value of column,
                  Split for minimum Gini
    
        '''
        self.listcol = [item[col] for item in self.data]
        self.keys = sorted(range(len(self.listcol)), key=lambda k: self.listcol[k])
        self.listcol.sort()
        min_gini_val = float('inf')
        split=0
        for k in range(1,self.rows):
            lsize = k
            rsize = self.rows - k
            lp = 0
            rp = 0
            for l in range(self.rows):
                if (self.labels[self.keys[l]] == 1) and l<k:
                    lp += 1
                if (self.labels[self.keys[l]] == 1) and l>=k:
                    rp += 1
            # Gini Calculation
            gini = (lsize / self.rows) * (lp / lsize) * (1 - lp / lsize) + (rsize / self.rows) * (rp / rsize) * (1 - rp / rsize)
            if min_gini_val > gini:
                min_gini_val = gini
                split=k
        return min_gini_val,split

    def column_selection(self):
    
        '''

        Function: Selecting Column with least Gini
        
        Input:
        Output:   Overall minimum Gini value,
                  Split for minimum Gini,
                  Column with minimum Gini

        '''
        overall_min_gini=float('inf')
        col_split=0
        col_num=0
        for j in range(self.cols):
            gini_val,split_val=self.gini_cal(j)
            if overall_min_gini>gini_val:
                overall_min_gini=gini_val
                col_split=split_val
                if col_split!=0:
                    col_split=(self.listcol[col_split] + self.listcol[col_split - 1]) / 2
                # Column with min Gini
                col_num=j

        return overall_min_gini,col_split,col_num


