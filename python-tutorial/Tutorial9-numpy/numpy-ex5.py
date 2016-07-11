from numpy import *
#a = array([1,2,3])
#print a.prod() # 1 * 2 * 3 = 6
b = array([[1,2,3],[4,5,6]])
print b
#print b.prod(dtype=float) # specify type of output
print b.prod(axis=0) # for each of the 3 columns: product
# array([ 4, 10, 18])
print b.prod(axis=1) # for each of the two rows: product
#array([ 6, 120])
print b.prod(axis=-1)
print b.prod(axis=-2)