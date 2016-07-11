#Basic operations examples
from numpy import *
a = array( [20,30,40,50] )
b = arange( 4 )
print  a-b
print  b**2
print 10*sin(a)
print a<35

A = array( [[1,1],[0,1]] )
B = array( [[2,0],[3,4]] )
# element wise product
print A*B    
# matrix product 
print dot(A,B)