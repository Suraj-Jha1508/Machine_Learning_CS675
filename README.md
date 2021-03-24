# Machine_Learning_CS675

Machine Learning Assignments and Project which i have done in my Masters at NJIT under Course CS675

#Assignment 1:

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

#Assignment 2:

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

#Assignment 3:

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

#Assignment 4:

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

#Assignment 5:

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

#Assignment 6:

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

#Assignment 7:
Write a Python program to perform bagging on the decision 
stump that you wrote in assignment 6.

The input should be the data file and labels as in previous
assignments. The output is the prediction of test datapoints just
like we did in assignments one through five. 

Your program will create a bootstrapped dataset and then run
your decision stump on it and obtain predictions labels.
It will repeat this a 100 times and output the majority vote of 
the predictions. 

#Assignment 8:

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

