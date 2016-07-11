from xml.dom.minidom import parseString

dom = parseString('<title></title>')
root = dom.documentElement
nelem = dom.createElement("separator")

root.insertBefore(nelem, None)
cdata = dom.createTextNode("The New Avengers")
root.insertBefore(cdata, nelem)
print root.toxml()
