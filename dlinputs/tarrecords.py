# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import os
import os.path
import StringIO
import tarfile
import warnings
import re
import time
import getpass
import socket

import utils

def splitallext(path):
    """Helper method that splits off all extension.

    Returns base, allext.

    :param path: path with extensions
    :returns: path with all extensions removed

    """
    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)

def base_plus_ext(fname):
    """Splits pathnames into the file basename plus the extension."""
    return splitallext(fname)

def dir_plus_file(fname):
    """Splits pathnames into the dirname plus the filename."""
    return os.path.split(fname)

def last_dir(fname):
    """Splits pathnames into the last dir plus the filename."""
    dirname, plain = os.path.split(fname)
    prefix, last = os.path.split(dirname)
    return last, plain

def tariterator(fileobj, check_sorted=False, keys=base_plus_ext, decode=True):
    """Iterate over samples from a tar archive, either locally or given by URL.

    Tar archives are assumed to be sorted by file name. For each basename,
    reads all the files with different extensions and returns a dictionary
    with the extension as key and the file contents as value.

    :param str archive: tar archive with sorted file names (file name or URL)
    :param bool check_sorted: verify that file names are sorted
    :returns: iterator over samples

    """
    if decode is True:
        decode = utils.autodecode
    elif decode is False:
        decode = lambda x: x
    current_count = 0
    current_prefix = None
    current_sample = None
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        if not tarinfo.isreg():
            continue
        fname = tarinfo.name
        if fname is None:
            warnings.warn("tarinfo.name is None")
            continue
        prefix, suffix = keys(fname)
        if prefix is None:
            warnings.warn("prefix is None for: %s" % (tarinfo.name,))
            continue
        if prefix != current_prefix:
            if check_sorted and prefix <= current_prefix:
                raise ValueError("[%s] -> [%s]: tar file does not contain sorted keys" % \
                                 (current_prefix, prefix))
            if current_sample is not None and \
               not current_sample.get("__bad__", False):
                yield decode(current_sample)
            current_prefix = prefix
            current_sample = dict(__key__=prefix)
        try:
            data = stream.extractfile(tarinfo).read()
        except tarfile.ReadError, e:
            print "tarfile.ReadError at", current_count
            print "file:", tarinfo.name
            print e
            current_sample["__bad__"] = True
        else:
            current_sample[suffix] = data
            current_count += 1
    if len(current_sample.keys()) > 0:
        yield decode(current_sample)
    try: del stream
    except: pass

class TarWriter(object):
    def __init__(self, fileobj, keep_meta=False, encode=None, user=None, group=None):
        """A class for writing dictionaries to tar files.

        :param fileobj fileobj: file name for tar file (.tgz)
        :param bool keep_meta: keep fields starting with "_"
        :param function encoder: encoding of samples prior to writing
        """
        if encode is True:
            encode = utils.autoencode
        elif encode is False:
            encode = lambda x: x
        self.keep_meta = keep_meta
        self.encode = encode
        self.stream = None
        self.tarstream = tarfile.open(fileobj=fileobj, mode="w:gz")
        self.user = user or getpass.getuser()
        self.group = group or socket.gethostname()

    def __enter__(self):
        """Context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager."""
        self.finish()

    def close(self):
        """Close the tar file."""
        self.finish()

    def finish(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.stream:
            self.stream.close()
            self.stream = None

    def write(self, obj):
        """Write a dictionary to the tar file.

        :param str obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        obj = self.encode(obj)
        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if not self.keep_meta and k[0]=="_":
                continue
            v = obj[k]
            assert isinstance(v, (str, buffer)),  \
                "converter didn't yield a string: %s" % ((k, type(v)),)
            now = time.time()
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mtime = now
            ti.mode = 0o666
            ti.uname = "bigdata"
            ti.gname = "bigdata"
            stream = StringIO.StringIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total
