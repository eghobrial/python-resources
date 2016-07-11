from scipy import *
from numpy import *
import matplotlib.pyplot as plt

n = 20
random.seed(20)
x = linspace (0.0, 1.0, n)
noise = random.normal(0, 0.25, n)
a_e = -2.0
b_e = 3.0
y_line = a_e*x + b_e
y = y_line + noise

A = array ([x, zeros(n) + 1])
A = A.T


result = linalg.lstsq(A,y)
#print result
a, b = result[0]

plt.plot(x,y, 'o', x, y_line, 'r', x , a*x + b, 'b')

plt.legend(('data points', ' original line', 'fitted line'),
           'upper center', shadow=True, fancybox=True)
plt.title('y = %g*x + %g: fit to y = %g*x + %s + normal noise' % (a,b, a_e, b_e))
plt.show()
