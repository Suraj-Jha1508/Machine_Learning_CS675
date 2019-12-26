import numpy as np
from Time_Series_Analysis import LstmMethod

def mean_squared_error(prediction,test):
    sum=0
    n=len(prediction)
    for i in range(n):
        sum+=((prediction[i]-test[i])**2)
    return sum/n

print('Reading Data...')
data = np.genfromtxt('Sales_Transactions_Dataset_Weekly.csv', delimiter=',')
train_data = data[1:, 1:52]
test_data=data[1:, 52:53]
test_data=[item[-1] for item in test_data]
window=51
product_num,week_num=np.shape(train_data)
print('Dimension of train data: {} X {}'.format(product_num,week_num))
print('Applying LSTM and Regression...')
min_error=float('inf')
min_w=0
min_error_pred_data=[]
for w in range(2,window):
    pred_data=[]
    for i in range(product_num):
        object=LstmMethod(train_data[i],w)
        pred_data.append(object.ridgeRegression())
    error=mean_squared_error(pred_data,test_data)
    print('Mean Square Error:{} for window:{}'.format(error,w))
    if error<min_error:
        min_error=error
        min_error_pred_data=pred_data
        min_w=w

for i in range(len(min_error_pred_data)):
    print('{}\t{}'.format(round(min_error_pred_data[i][0],1),(i+1)))
print('Minimum Mean Square Error:{} for window:{}'.format(round(min_error[0],2),min_w))







