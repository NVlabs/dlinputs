#!/usr/bin/python

import os
import imp
import simplejson

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

def load_py_input(iname, modname="inlib"):
    """Load an input and return a module containing an input pipeline.

    :param iname: input module (.py), with optional key=value pairs
    :param modname: name for the loaded module
    :returns: instance of modname.Inputs(**params)

    """
    iname, params = get_params(iname)
    inlib = imp.load_source(modname, iname)
    return inlib.Inputs(**params)

def load_input(iname, method="default"):
    parsed = urlparse.urlparse(iname)
    if parsed.scheme.startswith("z"):
        import zcom
        return zcom.Connection(iname).items()
    elif parsed.path.endswith(".py"):
        inputs = dli.loadable.load_input(iname)
        datasets = [attr[:-5] for attr in sorted(dir(inputs)) if attr.endswith("_data")]
        print "datasets:", ", ".join(datasets)
        print "showing:", args.table
        print
        method = "{}_data".format(method)
        assert hasattr(inputs, method), \
            "{} does not define {}_data method".format(iname)
        return getattr(inputs, method)()
    elif parsed.scheme.endswith(".tgz"):
    else:
        raise Exception("{}: unknown extension".format(iname))
