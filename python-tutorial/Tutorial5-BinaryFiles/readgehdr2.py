import struct
import os, mmap

filename='P09728_spepgj_111104_0938.7'
f=open(filename,'rb')
f_size=os.path.getsize(filename)
fdata=mmap.mmap(f.fileno(),f_size,access=mmap.ACCESS_READ)
subset = fdata[0:16]
s = struct.Struct('f i h 6s')
var = s.unpack(subset)
print var