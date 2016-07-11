from xml.dom.minidom import parse
# Create a function to get the data between XML tags
# Information is held in nodes (discrete blocks)
# which we�ll need to concatenate together to get the
# full picture. We only need to add text nodes to the
# string
def getdata(nodes):
    rc = ''
    for node in nodes:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc

# Parse the document
client = parse('client.xml')

# Get the first clientname tag from the document
clientname = client.getElementsByTagName("clientname")[0]

# Print out the data contained within the tags
# using getdata to extract the text from the nodes
# defined within the element
print 'Client name is', getdata(clientname.childNodes)