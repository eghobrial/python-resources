from xml.dom.minidom import parse
import matplotlib.pylab as plt
from numpy import *
#from dateutil.parser import parse

def getdata(nodes):
    rc = ''
    for node in nodes:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc

def handlegroup(temp):
    group = temp.getElementsByTagName("group")[0]
    parameters = group.getElementsByTagName("parameter")
    handleparameters(parameters)
    
def handleparameters(parameters):
    #print 'Parameters:'
    dofBoreRawT = {}
    dofGRawT = {}
    for parameter in parameters:
        parattrib1 = parameter.getAttribute("name")
        parattrib2 = parameter.getAttribute("unit")  
        #print 'Name ', parattrib1, ' Unit ' , parattrib2
        if parattrib1 == "Bore Raw Temp":
            dofBoreRawT = handlevalueBRT(parameter,dofBoreRawT) 
            #print dofBoreRawT
        if parattrib1 == "Gradient Raw Temp":
            dofGRawT = handlevalueGRT(parameter,dofGRawT)
    #print dofGRawT   
    plotvalues(dofGRawT)
            
def plotvalues(dofGRawT):
    #times = sorted(dofGRawT.keys())
    #values = dofGRawT.values()
    items = dofGRawT.items()
    print len(items)
    y1 = zeros(len(items))
    y2 = zeros(len(items))
    y3 = zeros(len(items))
    y4 = zeros(len(items))
    time = zeros(len(items))
    for index in range (0,len(items)):
        #time[index] = items[index][0] 
        #ts = parse(items[index][0])
        #print ts
        item = items[index]
        temps = item[1].split(',')
        #print item[0]
        #print temps
        y1[index] = int(temps[0])
        y2[index] = int(temps[1])
        y3[index] = int(temps[2])
        y4[index] = int(temps[3])
    #print y2   
    #print time
    plt.xlabel('Temp')
    plt.ylabel('some numbers')
    plt.plot(y1,'r--',y2,'g--',y3,'b--',y4,'m--')
    plt.show() 
    
def handlevalueBRT(parameter,dofBoreRawT):
    value =  parameter.getElementsByTagName("value")[0]
    tempattrib1 = value.getAttribute("size")
    tempattrib2 = value.getAttribute("starttime")
    #print 'Size ', tempattrib1, ' start time ' , tempattrib2, ' Temp Value ', getdata(value.childNodes)
    #len(dofBoreRawT)
    dofBoreRawT[tempattrib2]=getdata(value.childNodes)
    return dofBoreRawT

def handlevalueGRT(parameter,dofGRawT):
    value =  parameter.getElementsByTagName("value")[0]
    tempattrib1 = value.getAttribute("size")
    tempattrib2 = value.getAttribute("starttime")
    #print 'Size ', tempattrib1, ' start time ' , tempattrib2, ' Temp Value ', getdata(value.childNodes)
    #len(dofGRawT)
    dofGRawT[tempattrib2]=getdata(value.childNodes)
    return dofGRawT
#def handlevalue(parameter):
#    value =  parameter.getElementsByTagName("value")[0]
#    tempattrib1 = value.getAttribute("size")
#    tempattrib2 = value.getAttribute("starttime")
#    print 'Size ', tempattrib1, ' start time ' , tempattrib2, ' Temp Value ', getdata(value.childNodes)
    #print 'start time' + tempattrib

# Parse the document
temp = parse('sriTemp.xml')
handlegroup(temp)

#tempvalue = temp.getElementsByTagName("value")[0]
#tempattrib = tempvalue.getAttribute("starttime")
#print 'start time' + tempattrib

# Print out the data contained within the tags
# using getdata to extract the text from the nodes
# defined within the element
#print 'Parameter', getdata(tempvalue.childNodes)

