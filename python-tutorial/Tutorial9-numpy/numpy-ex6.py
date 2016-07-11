from numpy import *
a = array([1,2,7])
print a.mean()
b = array([[1,2,7],[4,9,6],[1,2,7],[4,9,6]])
print b
print b.mean()
print b.mean(axis=0) # the mean of each of the 3 columns
#array([ 2.5, 5.5, 6.5])
print b.mean(axis=1) # the mean of each of the 2 rows
#array([ 3.33333333, 6.33333333])
print b.mean(axis=-1)
print b.mean(axis=-2)

a = array([[1.,2,7],[4,9,6]])
print a.std()
print a.std(axis=0) # standard deviation of each of the 3 columns
#array([ 1.5, 3.5, 0.5])
print a.std(axis=1) # standard deviation of each of the 2 columns
#array([ 2.62466929, 2.05480467])