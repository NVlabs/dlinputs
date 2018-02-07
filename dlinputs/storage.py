import sys
import os
import os.path
import re
import StringIO
import warnings
import tempfile
from contextlib import closing
from subprocess import Popen, PIPE, check_call

class ObjectStorage(object):
    def __init__(self):
        self.pattern = "unimplemented:.*"
    def open_read(self, location):
        raise Exception("unimplemented")
    def open_write(self, location):
        raise Exception("unimplemented")
    def delete(self, location):
        raise Exception("unimplemented")
    def rename(self, old, new):
        raise Exception("unimplemented")
    def list(self, location):
        raise Exception("unimplemented")
    def exists(self, location):
        try:
            with self.open_read(location) as stream:
                pass
            return True
        except:
            return False

class GsStorage(ObjectStorage):
    """Open streams on Google Cloud object storage.

    This uses the `gsutil` command line tool, both for simplicity
    and because it allows asynchronouse input/output operations."""
    def __init__(self):
        self.pattern = "^gs:.*$"
        assert os.system("gsutil -v > /dev/null 2>&1") == 0, "gsutil not working"
    def open_read(self, location):
        return os.popen("gsutil cat '%s'" % location, "rb")
    def open_write(self, location):
        return os.popen("gsutil cp - '%s'" % location, "wb")

class HttpStorage(ObjectStorage):
    """Open streams on web servers.

    This uses `curl` for asynchronous I/O. You can also use
    a `.netrc` to handle authentication with `curl`."""
    def __init__(self, pretest=True):
        self.pattern = "^https?:.*$"
        self.pretest = pretest
        assert os.system("curl --version > /dev/null 2>&1") == 0, "curl not working"
    def test_location(self, location):
        proc = Popen(["curl", "--fail", "-s", "-T", "-", location], stdin=PIPE)
        proc.stdin.close()
        if proc.wait() != 0:
            raise Exception("{}: cannot write location".format(location))
        check_call(["curl", "--fail", "-X", "DELETE", location])
    def open_read(self, location):
        cmd = "curl --fail -s '%s'" % location
        return os.popen(cmd, "rb")
    def open_write(self, location):
        if self.pretest:
            self.test_location(location)
        cmd = "curl --fail -s -T - '%s'" % location
        return os.popen(cmd, "wb")

class LocalStorage(ObjectStorage):
    """Open streams on the local file system."""
    def __init__(self):
        self.pattern = "^[^:]*$"
    def open_read(self, location):
        return open(location, "rb")
    def open_write(self, location):
        return open(location, "wb")

class GenericStorage(ObjectStorage):
    """Open I/O streams based on URL patterns."""
    def __init__(self):
        self.handlers = []
        for constructor in [GsStorage, HttpStorage, LocalStorage]:
            try:
                self.handlers += [constructor()]
            except Exception, _:
                print "{}: failed to initialize".format(constructor)
    def find_handler(self, location):
        for handler in self.handlers:
            if re.search(handler.pattern, location):
                # print location, handler
                return handler
        raise Exception("{}: no storage handler".format(location))
    def open_read(self, location):
        return self.find_handler(location).open_read(location)
    def open_write(self, location):
        return self.find_handler(location).open_write(location)

storage = GenericStorage()
generic = storage
