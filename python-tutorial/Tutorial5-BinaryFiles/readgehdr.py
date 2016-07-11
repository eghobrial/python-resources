import struct
f=open('P09728_spepgj_111104_0938.7')
fdata=f.read()
type(fdata) # returns <type 'str'>
len(fdata)  #returns 7313967388
struct.unpack('f',fdata[0:4]) #returns (20.006000518798828,)
struct.unpack('i',fdata[4:8]) #returns (9728,)
struct.unpack('h',fdata[8:10] #returns (14,)