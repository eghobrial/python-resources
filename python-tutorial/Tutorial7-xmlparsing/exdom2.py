from xml.dom.minidom import parse

def getdata(nodes):
    rc = ''
    for node in nodes:
        if node.nodeType == node.TEXT_NODE:
            rc = rc + node.data
    return rc

def handleclient(client):
    clientname = client.getElementsByTagName("clientname")[0]
    print 'Client:', getdata(clientname.childNodes)
    accounts = client.getElementsByTagName("account")
    handleaccounts(accounts)

def handleaccounts(accounts):
    print 'Accounts:'
    for account in accounts:
        handleaccount(account)

def handleaccount(account):
    accname = account.getElementsByTagName("accname")[0]
    provider = account.getElementsByTagName("provider")[0]
    print ' ' * 4, '%s (%s)' % (getdata(accname.childNodes),
                                getdata(provider.childNodes))
    print ' ' * 4, 'Transactions:'
    trans = account.getElementsByTagName("transaction")
    for transaction in trans:
        handletransaction(transaction)
    balance = account.getElementsByTagName("balance")[0]
    print ' ' * 9, '%-40s %s' % ('', '======')
    print ' ' * 9, '%-40s %s' % ('', getdata(balance.childNodes))
    print ''
def handletransaction(transaction):
    payee = transaction.getElementsByTagName("payee")[0]
    amount = transaction.getElementsByTagName("amount")[0]
    print ' ' * 9, '%-40s %s' % (getdata(payee.childNodes),
                                 getdata(amount.childNodes))

client = parse('client.xml')

handleclient(client)