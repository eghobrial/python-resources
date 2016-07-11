from numpy import *
x = array([0,1,3,9,5,10])
print x
print 'first order diff of x ', diff(x) # 1st-order differences between the elements of x
print 'second order diff of x ', diff(x,n=2) # 2nd-order differences, equivalent to diff(diff(x))
a = array([[1,3,6,10],[0,5,6,8]])
print a
print 'first order diff of a between col ', diff(a) # 1st-order differences between the columns (default: axis=-1)
print 'first order diff of a between col with -1 ', diff(a,axis=-1) # 1st-order differences between the columns (default: axis=-1)
print 'first order diff of a between raws ', diff(a,axis=0) # 1st-order difference between the rows

y = array([1,2,3]) # cumulative sum = intermediate summing results & total sum
print 'intermediate suming for array y = array([1,2,3]) ', y.cumsum() # note that cumsum(y) also exists
a = array([[1,2,3],[4,5,6]])
print 'Original array ', a
print 'specify float as a type for cumsum', a.cumsum(dtype=float) # specifies type of output value(s)
print 'sum across columns', a.cumsum(axis=0) # sum over rows for each of the 3 columns
print 'sum across rows', a.cumsum(axis=1) # sum over columns for each of the 2 rows