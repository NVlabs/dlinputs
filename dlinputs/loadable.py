#!/usr/bin/python

import os
import imp
import glob
import torch
import simplejson

def _convert(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def get_params(fname):
    """Splits a file name into the actual file and optional parameters.

    Filenames may be of the form `foo:x=y:a=b`, and this returns 
    `(foo, dict(x="y", a="b")`
    """
    params = fname.split(":")
    fname = params[0]
    params = params[1:]
    params = [p.split("=") for p in params]
    params = {k: _convert(v) for k, v in params}
    return fname, params

def load_input(iname, modname="inlib"):
    """Load an input and return a module containing a get_training /
    get_testing function."""
    iname, params = get_params(iname)
    inlib = imp.load_source(modname, iname)
    return inlib.Inputs(**params)

def make_meta():
    return dict(ntrain=0,
                logging=[],
                params=[],
                training_loss=[],
                test_loss=[])

def load_model(mname, modname="modlib"):
    mname, params = get_params(mname)
    if not mname.endswith(".py"):
        return None
    modlib = imp.load_source(modname, mname)
    result = modlib.Model(**params)
    result.META = make_meta()
    return result

def load_net(mname, mparams={}):
    """Load a model, either a module containing a get_model function
    or a saved torch model. For saved models, also returns metadata."""
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
    """Save a model and model metadata."""
    ext = ".lock"
    torch.save(mname+ext, model)
    os.link(mname+ext, mname)
    # we save metadata as a JSON file as well to allow fast indexing / search
    if hasattr(model, "META"):
        simplejson.save(mname+".json", model.META)
    os.unlink(mname+ext, mname)
