from numpy import *
#creates an array with values between 0-9 and reshape into a 2 rows by 5 col
# rank 2 array
a = arange(10).reshape(2,5)
#ndarray.ndim, number of axes, rank
print 'ndim  ' , a.ndim
#ndarray.shape, size of array in each dim
print 'shape  ' ,  a.shape
#ndarray.size, total number of elements
print 'size  ' , a.size
#ndarray.dtype, item type ex int32
print 'dtype ' , a.dtype
#ndarray.itemsize, print the # of bytes/item
print 'itemsize ' , a.itemsize
#ndarray.data, the data buffer normaly not used
print 'data ' , a.data