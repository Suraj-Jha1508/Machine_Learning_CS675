# Machine_Learning_CS675

This Repository contain all Assignments and Project i completed during my Machine Learning Course at NJIT



# Assignment 1: Naive Bayes classifier

Write a Python program that implements the Naive Bayes classifier.
Your program should take as input a dataset file and a set of training
labels in the format given in the example datasets on the course website.
As output your program should produce predicted labels for the test
dataset which are feature vectors whose labels are not given for training. 

To avoid divide by zero error that will occur when the variance is zero,
we use the pseudocount method. In this method we initialize the mean vector
to be some small value. So instead of initializing the mean to zero
we set it to 0.1 for example. See the document "Naive Bayes variance pseudocount"
in our course google drive.

# Assignment 2: Gradient descent for minimizing the least squares loss

Write a Python program that implements gradient descent for minimizing
the least squares loss. As a stopping condition check for the objective
between the current and previous iteration. If the objective improves
by less than theta then you stop. The input and output should be the same 
as for nearest means and Naive-Bayes. 

Test your program with the input data

0 0
0 1
1 0
1 1
10 10
10 11
11 10
11 11

and labels 

0 0
0 1
0 2
0 3
1 4
1 5
1 6
1 7

Use eta=.001 and stopping condition of .001. 

Your final w would be close to

w = 0.0889184232356005 0.0907934047968894 

and distance of plane to origin would be about

abs(w0/||w||) = 7.09045903042441

If you change the stopping condition to 0, in other words 
full convergence, then your final w would be

w = 0.0995024069539168 0.0995025677420564 

and distance to origin 

abs(w0/||w||) = 7.77817457926694

# Assignment 3: Optimizing the SVM hinge loss

Write a Python program for optimizing the SVM hinge loss. 
descent algorithm. The input and output should be the same as for
assignment 2.

Test your program with the input data

1 1
1 2
1 3
3 1
3 2
3 3
50 2

and labels <to be provided>

0 0
0 1
0 2
1 3
1 4
1 5
1 6

Convert label 0 to -1 so that labels yi are either +1 or -1. This is
necessary for the gradient descent to work.

Use eta=.001 and stopping condition of while(abs(prevobj - obj) > .000000001). 
Note the absolute value to account for instability in the gradient for hinge 
loss. The converged solution with the hinge loss would be

w = (1.4605574252399243, -0.4595542036671061)
w0 = -2.0024682128830427
Dist to origin= 1.3078203832146862

# Assignment 4: Logistic discrimination gradient descent algorithm

Write a Python program for the logistic discrimination gradient
descent algorithm. The input and output should be the same as for nearest 
means and Naive-Bayes. 

Test your program with the input data

0 0
0 1
1 0
1 1
10 10
10 11
11 10
11 11

and labels

0 0
0 1
0 2
0 3
1 4
1 5
1 6
1 7

Do not convert 0 to -1 in the labels. They must remain 0 for the logistic 
regression gradient descent.

Use eta=.01 and stopping condition of .0000001. 

Your final w would be close to the one shown below. Note its similarity to
the perceptron output.

w = 0.957672135162093 0.956767618860693 
||w||=1.35371348333622
distance to origin = -6.83744723331703

You may also try the data 

1 2
2 1
2 2
2 3
4 1
4 2
4 3
50 2

and labels 

0 0
0 1
0 2
0 3
1 4
1 5
1 6
1 7

For this example the output would be similar to the one below

w = 6.77850714487713 -1.06370810572314 
||w||=6.86146005215591
distance to origin = -2.60844880003425

# Assignment 5: Adaptive eta setting Algorithm

Modify your solution for assignments 2 and 3 to do an adaptive
eta setting. Between the compute dellf and updatew code portions
insert the following pseudocode (see least squares in Perl code on
course website for reference). 

eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
bestobj = 1000000000000
for k in range(0, len(eta_list), 1):

  eta = eta_list[k]
  
  ##update w
  ##insert code here for w = w + eta*dellf

  ##get new error
  error = 0
  for i in range(0, rows, 1):
    if(trainlabels.get(i) != None):
      ##update error
      ##insert code to update the loss (which we call error here)

  obj = error

  ##update bestobj and best_eta
  ##insert code here

  ##remove the eta for the next
  ##insert code here for w = w - eta*dellf

eta = best_eta

After you have the adapative step size solutions working obtain
the average test error of least squares and hinge on the six
datasets on the course website. For this use the avg_test_error
script from https://web.njit.edu/~usman/courses/cs675_summer19/avg_test_error.pl. 

Your assignment is due before midnight October 29th, 2019

Submit your programs as least_squares_adaptive_eta.py and 
hinge_adaptive_eta.py. Also submit your avg_test_error.pl script 
so that we can evaluate your programs on the six datasets. Don't
forget to copy the six datasets also into your course directory.

# Assignment 6: CART decision tree algorithm

Write a Python program that determines the column with the
best split for the CART decision tree algorithm. You don't
have to write the CART algorithm in its entirety. You just
have to write a program that will traverse all columns in the
data and output the column and the threshold that gives the
lowest gini index.

The input should be the data file and labels as in previous
assignments. The output is the column number k and the split
value s.

We will test it on a simple example to determine if your program 
gives the correct output. Test your program with a simple XOR example.

Your completed script is due midnight on July 28th 2019. 

High level pseudocode:

    (1) For each column j:
	        
	(1) Find the value that gives the minimum gini split of 
	the data d into a partition of two sets

	(2) To evaluate the gini of a split use the formula

	gini = (lsize/rows)*(lp/lsize)*(1 - lp/lsize) + (rsize/rows)*(rp/rsize)*(1 - rp/rsize);

	where lsize is the size of the left partition, lp is the 
	proportion of -1 labels in the left partition, rsize is the 
	size of the right partition, rp is the proportion of -1 
	labels in the right partition, and rows is the total number of
	datapoints in the dataset d (passed to the function)

    (2) Let column k give the best split s. Output k and s.

# Assignment 7: Bagging on the decision stump

Write a Python program to perform bagging on the decision 
stump that you wrote in assignment 6.

The input should be the data file and labels as in previous
assignments. The output is the prediction of test datapoints just
like we did in assignments one through five. 

Your program will create a bootstrapped dataset and then run
your decision stump on it and obtain predictions labels.
It will repeat this a 100 times and output the majority vote of 
the predictions. 

# Assignment 8: K-means clustering

Write a Python program to output a k-means clustering. Your program
would have similar structure to the nearest means program. Follow
the pseudocode given in the course slides. 

The input to your program is a dataset and number of cluster k.
The output is in the same format of label files we have been using
in the course. So if the clustering is C0 = {0, 2, 3}, C1 = {1, 4}
and C2 = {5} then the output would be

0 0
1 1
0 2
0 3
1 4
2 5

# Project 1: Feature Selection

In this course project we encourage you to develop your own set of methods 
for learning and classifying. 

We will test your program on the dataset provided for the project. This is 
a simulated dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features). Amongst all SNPs are 15 causal 
ones which means they and neighboring ones discriminate between case and 
controls while remainder are noise.

In the training are 4000 cases and 4000 controls. Your task is to predict 
the labels of 2000 test individuals whose true labels are known only to 
the instructor and TA. 

Both datasets and labels are immediately following the link for this
project file. The training dataset is called traindata.gz (in gzipped
format), training labels are in trueclass, and test dataset is called
testdata.gz (also in gzipped format).

You may use cross-validation to evaluate the accuracy of your method and for 
parameter estimation. The winner would have the highest accuracy in the test 
set with the fewest number of features.

Your project must be in Python. You cannot use numpy or scipy except for numpy 
arrays as given below. You may use the support vector machine, logistic regression, 
naive bayes, linear regression and dimensionality reduction modules but not the 
feature selection ones. These classes are available by importing the respective 
module. For example to use svm we do

from sklearn import svm

You may also make system calls to external C programs for classification
such as svmlight, liblinear, fest, and bmrm.

Memory issues:

One challenge with this project is the size of the data and loading it into 
RAM. Floats and numbers take up more than 4 bytes in Python because 
everything is really an object (a struct in C) that contain other 
information besides the value of the number. To reduce the space we can use 
the array class of Python.

Type 

from array import array

in the beginning of your program. Suppose we have a list of n float called 
l. This will take more space than 4l bytes. To make it space efficient 
create a new array called l2 = array('f', l). The new array l2 can be 
treated pretty much like a normal list except that it will take 4l bytes (as 
is done in C or C++).

You may also use numpy arrays for efficient storage.

Your program would take as input the training dataset, the 
trueclass label file for training points, and the test dataset. 
The output would be a prediction of the labels of the test dataset in the 
same format as in the class assignments. Also output the total number of 
features and the feature column numbers that were used for final predicton. 
If all features were used just say "ALL" instead of listing all column 
numbers.

The score of your output is measured by accuracy/(#number of features). 
In order to qualify for full points you would need to achieve an accuracy
of at least 63%.

# Project 2: generate Hyperplane for classification 

In this optional assignment we will experiment with random hyperplanes
for classification. Your program will take a dataset as input and
produce new features following the procedure below. The input is in
the same format as for previous assignments.

Input data matrix X: n rows, m columns
Input training labels Y
Input value of k

For i = 0 to k do:
	a. Create random vector w where each wj is uniformly sampled between -1 and 1.
	
	b. Let xj be our training data points. Determine the largest and smallest wTxj
	across all xj. Select w0 randomly between [smallest wTxj, largest wTxj].

	c. Project training data X (each row is datapoint xj) onto w. 
	Let projection vector zi be Xw + w0 (here X has dimensions n by m and w is m by 1).
	Append (1+sign(zi))/2 as new column to the right end of Z. Remember that zi is
	a vector and so (1+sign(zi))/2 is 0 if the sign is -1 and 1 otherwise.
	
	d. Project test data X' (each row is datapoint xj) onto w. 
	Let projection vector z'i be X'w + w0. Append (1+sign(z'i))/2 as new column to 
	the right end of Z'. We create the test data in exactly the same way as we do
	the training except that we do it on X' the test data instead of X the training data.
	
1. Run linear SVM on Z and predict on Z'
2. Do values of k=10, 100, 1000, and 10000.
3. How does the error compare to liblinear on original data X and X' for each k?

Submit a document containing the error of linear SVM (cross-validated C) on the 
first split of each of the six datasets on the course website. Do this on the original 
data representation and the new representation for all values of k.

Submit your program that creates features and run LinearSVC (in Python scikit)
on the new training data and predicts on the new test data. In LinearSVC set the
max_iter parameter to 10000 so that we do a deep search.

The input to your program is the same as for the assignments: the full dataset
plus the train labels and in addition a value k for the number of new features.
The output of your program should be the prediction of the test data with
cross-validated svm (for example LinearSVC) on the new data representation.

# Project 3: Time Series Prediction By LSTM method

The weekly sales transaction dataset (posted here 
https://web.njit.edu/~usman/courses/cs675_fall19/Sales_Transactions_Dataset_Weekly.csv) 
shows weekly sales of over 800 items across a year. Your task is to predict the 
final week's sales from the previous values for each item in the dataset. Report 
your mean squared error which is defined as the mean squared error of your 
predictions 1/n(sum_i (y'i - yi)**2). The best mean squared that we achieve in this 
dataset is about 17.5 with ridge regression applied to an LSTM encoding of the data.

You may use numpy, sklearn, and pandas in your solution. Your program should 
consider the last week 51 as the test data and prior weeks as training.

Submit your program that takes as input the dataset 
Sales_Transactions_Dataset_Weekly.csv and outputs the predictions of week 51 for 
each item and the mean squared error.

It's very important that your code does not consider the last column during training.
If it does we will have to assign a grade of 0. If the code is too complicated to
decipher and we cannot tell if you consider the last column we have no choice but
to assign a 0. 

To avoid such problems make it very clear in your code (with comments) that you
are considering the data only up to week 51 in the training and that week 52 is
clearly specified as test.
