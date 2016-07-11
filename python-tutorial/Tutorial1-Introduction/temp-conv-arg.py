# import modules used here -- sys is a very standard one
import sys

# Gather our code in a main() function
def main():
  f = int(sys.argv[1])
  c = 5 * (f-32) / 9
  print "Temp in Farenheit = (%d), equivelant in Celsius = (%d)" % (f,c)  
 

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()