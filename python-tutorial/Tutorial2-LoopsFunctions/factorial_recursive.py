# define factorial function (Recursive method)
def factorial (n):
  if n == 1:
    return 1
  else:
    return n * factorial(n-1)
 
user_input = int (raw_input("Enter number to calculate factorial of "))
factorial_of_input = factorial(user_input)
print "Factorial of %d is equal to %d" % (user_input, factorial_of_input)  
  