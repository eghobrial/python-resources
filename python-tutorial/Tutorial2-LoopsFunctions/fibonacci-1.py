#!/usr/bin/python -tt

def fibonacci(N):
  if N==0:
    return 0
  elif N==1:
    return 1
  else:
    return fibonacci(N-1)+fibonacci(N-2)

user_input = input ("Enter a non-negative number to compute the fibonacci for ")
fibonacci_of_input = fibonacci(user_input)
print fibonacci_of_input