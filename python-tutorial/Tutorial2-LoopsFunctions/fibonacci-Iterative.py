#!/usr/bin/python -tt

def fibonacci(N):
  flist = [0,1]
  i = 2
  while i <= N:
    flist.append(flist[i-1]+flist[i-2])
    i = i +1
  return flist[N]

user_input = input ("Enter a non-negative number to compute the fibonacci for ")
fibonacci_of_input = fibonacci(user_input)
print fibonacci_of_input