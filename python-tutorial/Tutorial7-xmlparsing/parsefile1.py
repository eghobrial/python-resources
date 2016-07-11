from xml.dom.minidom import parse, parseString

# Parse a bare string as XML

stringdoc = parseString('<para>Some text</para>')

# Parse a file object

xmlfile1 = open('myfile.xml')
filedoc = parse(xmlfile1)

# Parse a file directly

filedoc = parse('myfile.xml')