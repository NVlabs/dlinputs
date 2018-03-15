import msgpack
import zmq
import inputs
from urllib2 import urlparse
import logging

schemes = dict(
    # (KIND, BIND)
    zpush=(zmq.PUSH, False),
    zpull=(zmq.PULL, True),
    zpub=(zmq.PUB, True),
    zsub=(zmq.SUB, False),
    zrpush=(zmq.PUSH, True),
    zrpull=(zmq.PULL, False),
    zrpub=(zmq.PUB, False),
    zrsub=(zmq.SUB, True)
)

class Connection(object):
    def __init__(self, url, encode=True, pack=True):
        self.encode = encode
        self.pack = pack
        self.addr = urlparse.urlparse(url)
        kind, bind = schemes[self.addr.scheme]
        self.context = zmq.Context()
        self.socket = self.context.socket(kind)
        location = "tcp://"+self.addr.netloc
        if bind:
            print "bind", location
            self.socket.bind(location)
        else:
            print "connect", location
            self.socket.connect(location)
        if kind==zmq.SUB:
            self.socket.setsockopt(zmq.SUBSCRIBE, '')
    def send(self, data):
        if self.encode:
            data = inputs.autoencode(data)
        if self.pack:
            data = msgpack.dumps(data)
        self.socket.send(data)
    def recv(self):
        data = self.socket.recv()
        if self.pack:
            data = msgpack.loads(data)
        if self.encode:
            data = inputs.autodecode(data)
        return data
    def serve(self, source):
        for sample in source:
            self.send(sample)
    def items(self):
        while True:
            result = self.recv()
            yield result
