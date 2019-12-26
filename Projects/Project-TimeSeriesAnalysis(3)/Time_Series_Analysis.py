from sklearn.linear_model import Ridge

class LstmMethod():

    def __init__(self,train_data,window_size):
        self.window_size=window_size
        self.train_data=train_data
        self.data_size=len(train_data)
        self.new_train=[]
        self.new_labels=[]
        self.test_data=[[]]
        for i in range(self.data_size-self.window_size):
            x=[]
            for j in range(self.window_size):
                x.append(self.train_data[i+j])
            self.new_train.append(x)
            self.new_labels.append(self.train_data[i+self.window_size])
        for i in range(self.window_size):
            self.test_data[0].append(self.train_data[self.data_size-self.window_size+i])

    def ridgeRegression(self):
        clf = Ridge()
        clf.fit(self.new_train, self.new_labels)
        pred = clf.predict(self.test_data)
        return pred



