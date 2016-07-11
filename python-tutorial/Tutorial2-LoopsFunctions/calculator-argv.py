# import modules used here -- sys is a very standard one
import sys

# Gather our code in a main() function
def main():
  op1 = int(sys.argv[1])
  operator = sys.argv[2]  
  op2 = int(sys.argv[3])
  if operator == '+':
    result = op1+op2
    sucess = 1
  elif operator == '-':
    result = op1-op2
    sucess = 1
  elif operator == 'x':
    result = op1*op2
    sucess = 1
  elif operator == '/':
    result = op1/op2
    sucess = 1
  else:
    sucess = 0  
  if sucess:
    print "Result of %d (%s) %d = %d" % (op1,operator,op2,result)  
  else:
    print "Not valid operator"

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()