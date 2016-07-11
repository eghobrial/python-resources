from numpy import *
# to create an array use the command below
# note that omitting the square brackets is a common error
a = array ([2,3,4]) #default type is int32
print ' a = ', a
b = array ([1.2, 3.5, 5.1]) # forcing float 64 type
print ' b = ', b
c = array ([(1.5,2,3), (4,5,6) ])
print ' c = ', c
d = array ( [ [1,2], [3,4] ], dtype=complex )
print ' d = ', d
#set_printoptions(threshold='nan')
print arange(10000).reshape(100,100)