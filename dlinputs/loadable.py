#!/usr/bin/python

import urllib.parse

from future import standard_library

standard_library.install_aliases()

def _convert(x):
    """Generically convert strings to numbers.

    :param x: string that maybe represents a number
    :returns: value
    :rtype: string, float, or int

    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def get_params(fname, separator=":"):
    """Splits a file name into the actual file and optional parameters.

    Filenames may be of the form `foo:x=y:a=b`, and this returns 
    `(foo, dict(x="y", a="b")`

    :param fname: file name with optional key-value pairs
    :param separator: optional separator (default=":")

    """
    params = fname.split(separator)
    fname = params[0]
    params = params[1:]
    params = [p.split("=") for p in params]
    params = {k: _convert(v) for k, v in params}
    return fname, params
