# Load a file, based on the supplied filename
    # feeding the information to the XML parser
    def loadfile(self, filename):
        xmlfile = open(filename)
        while 1:
            data = xmlfile.read(1024)
            if not data:
                break
            self.feed(data)
        self.close()

    # Called when a start tag is found
    def unknown_starttag(self, tag, attrs):
        print "Start: ",tag, attrs

    # Called when an end tag is found
    def unknown_endtag(self, tag):
        print "End:   ",tag

    # Called when raw data is found
    def handle_data(self, data):
        print "Data:  ",data

# Get the first argument from the command line
try:
    filename = sys.argv[1]
except IndexError:
    print "You must supply a filename"
    sys.exit(1)

# Create a new MyParser instance and parse the
# file supplied on the command line
# We ignore EOFError’s, which just indicate the
# end of file
# The xmllib.Error exception is raised by xmllib’s
# parser when an error occurs
try:
    parser = MyParser(sys.argv[1])
except EOFError:
    pass
except xmllib.Error,data:
    print "There was an error in the XML:",data
    sys.exit(1)
except:
    print "Something went wrong"
    sys.exit(1)

print "Everything seems fine"

