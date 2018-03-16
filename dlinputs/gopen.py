import os
import sys
from urllib2 import urlparse
from contextlib import closing
from subprocess import Popen, PIPE, check_call
import random

import paths
import tarrecords

def test_curl_write(self, location):
    """Test whether we can write to a location using curl."""
    proc = Popen(["curl", "--fail", "-s", "-T", "-", location], stdin=PIPE)
    proc.stdin.close()
    if proc.wait() != 0:
        raise Exception("{}: cannot write location".format(location))
    check_call(["curl", "--fail", "-X", "DELETE", location])

def gopen(url, mode="rb"):
    """Open the given URL. Supports unusual schemes and uses subprocesses."""
    parsed = urlparse.urlparse(url)
    if parsed.scheme == "gs":
        if mode[0]=="r":
            return os.popen("gsutil cat '%s'" % url, "rb")
        elif mode[0]=="w":
            return os.popen("gsutil cp - '%s'" % url, "wb")
        else:
            raise ValueError("{}: unknown mode".format(mode))
    elif parsed.scheme in "http https ftp".split():
        if mode[0]=="r":
            cmd = "curl --fail -s '%s'" % url
            return os.popen(cmd, "rb")
        elif mode[0]=="w":
            self.test_location(url)
            cmd = "curl --fail -s -T - '%s'" % url
            return os.popen(cmd, "wb")
        else:
            raise ValueError("{}: unknown mode".format(mode))
    elif parsed.scheme in ["", "file"]:
        if mode[0]=="r":
            return open(parsed.path, "rb")
        elif mode[0]=="w":
            return open(parsed.path, "wb")
        else:
            raise ValueError("{}: unknown mode".format(mode))

def test_url(url, size=17):
    """Test whether the given URL is accessible."""
    try:
        with gopen(url) as stream:
            data = stream.read(size)
        if len(data) == size:
            return True
        return False
    except Exception, e:
        print e
        return False

def test_shards(url, size=17, complete=False):
    """Test whether the shard spec is accessible."""
    shards = list(paths.path_shards(url))
    if complete:
        return all(test_url(s, size=size) for s in shards)
    else:
        return test_url(shards[0], size=size)

def find_shards(urls, size=17, complete=False):
    """Given a list of shard URLs, find the first one that exists."""
    for url in urls:
        if test_shards(url, size=size, complete=False):
            return url

def sharditerator(url, epochs=1000000, shuffle=True, **kw):
    """Iterate over sharded tar records."""
    shards = list(paths.path_shards(url))
    for epoch in xrange(epochs):
        if shuffle: random.shuffle(shards)
        for shard in shards:
            with gopen(shard) as stream:
                for sample in tarrecords.tariterator(stream, **kw):
                    yield sample

def sharditerator_test(url, **kw):
    """Iterate over sharded tar records (no shuffling, one epoch only)."""
    return sharditerator(url, epochs=1, shuffle=False, **kw)
