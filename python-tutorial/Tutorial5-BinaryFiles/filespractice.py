import sys, math
try:
    infilename = sys.argv[1]
    outfilename = sys.argv[2]
except:
    print "Usage:", sys.argv[0], "infile outfile"
    sys.exit(1)
ifile = open (infilename, 'r')
outfile = open (outfilename, 'w')

def myfunc(y):
  if y >= 0.0:
      return y**5*math.exp(-y)
  else:
      return 0.0
  
for line in ifile:
    y = float(line)
    fy = myfunc(y)
    outfile.write('%g   %12.5e\n' % (y,fy))
    
ifile.close()
outfile.close()