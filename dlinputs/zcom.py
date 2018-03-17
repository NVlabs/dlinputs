import msgpack
import zmq
import inputs
from urllib2 import urlparse
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
    def __init__(self, url, codec=True, pack=True, stats_horizon=1000):
        if codec is False:
            codec = lambda x: x
        self.stats = Statistics(stats_horizon)
        self.codec = codec
        self.pack = pack
        self.addr = urlparse.urlparse(url)
        kind, bind = schemes[self.addr.scheme]
        self.context = zmq.Context()
        self.socket = self.context.socket(kind)
        location = "tcp://"+self.addr.netloc
        self.socket.setsockopt(zmq.LINGER, 0)
        if bind:
            self.socket.bind(location)
        else:
            self.socket.connect(location)
        if kind==zmq.SUB:
            self.socket.setsockopt(zmq.SUBSCRIBE, '')
    def close(self):
        self.socket.close()
        self.socket = None
        self.context = None
    def send(self, data):
        if self.codec is True:
            data = inputs.autoencode(data)
        else:
            data = self.codec(data)
        if self.pack:
            data = msgpack.dumps(data)
        self.socket.send(data)
        self.stats.add(len(data))
    def recv(self):
        data = self.socket.recv()
        self.stats.add(len(data))
        if self.pack:
            data = msgpack.loads(data)
        if self.codec is True:
            data = inputs.autodecode(data)
        else:
            data = self.codec(data)
        return data
    def serve(self, source):
        for sample in source:
            self.send(sample)
    def items(self):
        while True:
            result = self.recv()
            yield result
