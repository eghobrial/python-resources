# define factorial function (Iterative method)
def factorial (n):
  product = 1
  for i in range (n):
    product = product * (i+1)
  return product

user_input = int (raw_input("Enter number to calculate factorial of "))
factorial_of_input = factorial(user_input)
print "Factorial of %d is equal to %d" % (user_input, factorial_of_input)