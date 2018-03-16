import msgpack
import zmq
import inputs
from urllib2 import urlparse
import logging
import collections
import time

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

class Statistics(object):
    def __init__(self, horizon=1000):
        self.horizon = horizon
        self.reset()
    def reset(self):
        self.start = time.time()
        self.last = time.time()
        self.count = 0
        self.total = 0
        self.recent = collections.deque(maxlen=self.horizon)
    def add(self, x):
        self.last = time.time()
        self.count += 1
        self.total += x
        self.recent.append((self.last, x))
    def rate(self):
        if self.count==0: return 0
        return self.count / (self.last - self.start)
    def throughput(self):
        if self.count==0: return 0
        return self.total / (self.last - self.start)
    def recent_rate(self):
        if self.count==0: return 0 
        return len(self.recent) / (self.recent[-1][0] - self.recent[0][0])
    def recent_throughput(self):
        if self.count==0: return 0
        total = sum(r[1] for r  in self.recent)
        return total / (self.recent[-1][0] - self.recent[0][0])

class Connection(object):
    def __init__(self, url, encode=True, pack=True, stats_horizon=1000):
        self.stats = Statistics(stats_horizon)
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
        self.stats.add(len(data))
    def recv(self):
        data = self.socket.recv()
        self.stats.add(len(data))
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
