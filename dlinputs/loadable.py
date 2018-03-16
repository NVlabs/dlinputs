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

def load_input(iname, modname="inlib"):
    """Load an input and return a module containing an input pipeline.

    :param iname: input module (.py), with optional key=value pairs
    :param modname: name for the loaded module
    :returns: instance of modname.Inputs(**params)

    """
    iname, params = get_params(iname)
    inlib = imp.load_source(modname, iname)
    return inlib.Inputs(**params)

def make_meta():
    """Make initial model meta information dictionary.

    This contains information like `ntrain`, etc.

    :returns: dictionary of common model-related information.

    """
    return dict(ntrain=0,
                logging=[],
                params=[],
                training_loss=[],
                test_loss=[])

def load_model(mname, modname="modlib"):
    """Load a neural network model from a .py file.

    The model still needs to be created / instantiated by calling
    its create(...) method.

    :param mname: module name, with optional key=value pairs
    :param modname: module name to load module into
    :returns: instance of modname.Model(**params)

    """
    mname, params = get_params(mname)
    if not mname.endswith(".py"):
        return None
    modlib = imp.load_source(modname, mname)
    result = modlib.Model(**params)
    result.META = make_meta()
    with open(mname, "r") as stream:
        result.META["py_model"] = stream.read()
    result.META["py_params"] = params
    return result

def load_net(mname, mparams={}):
    """Load a model, either a module containing a get_model function

    :param mname: module name, either .pt or .py
    :param mparams: model parameters used for model.create(**params)

    """
    import torch
    model = load_model(mname)
    if model is not None:
        model = model.create(**mparams)
        return model
    else:
        model = torch.load(mname)
    if not hasattr(model, "META"):
        model.META = make_meta()
    return model

def save_net(mname, model):
    """Save a model, and separately also save its metadata.

    :param mname: file name for saving
    :param model: model to be saved

    """
    import torch
    ext = ".lock"
    torch.save(model, mname+ext)
    os.link(mname+ext, mname)
    # we save metadata as a JSON file as well to allow fast indexing / search
    if hasattr(model, "META"):
        with open(mname+".json", "w") as stream:
            simplejson.dump(model.META, stream)
    os.unlink(mname+ext)
