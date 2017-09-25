#!/usr/bin/python

import os
import imp
import glob
import torch

def _convert(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def get_params(fname):
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

def load_model(mname, modname="modlib"):
    mname, params = get_params(mname)
    if not mname.endswith(".py"):
        return None
    modlib = imp.load_source(modname, mname)
    return modlib.Model(**params)

def load_net(mname, mparams={}):
    """Load a model, either a module containing a get_model function
    or a saved torch model. For saved models, also returns metadata."""
    model = load_model(mname)
    if model is not None:
        model = model.create(**mparams)
        model_meta = dict(ntrain=0,
                          logging=[],
                          params=[],
                          training_loss=[],
                          test_loss=[])
        return model, model_meta
    model = torch.load(mname)
    model_meta = simplejson.load(mname+".json")
    return model, model_meta

def save_net(mname, model, model_meta):
    """Save a model and model metadata."""
    ext = ".lock"
    torch.save(mname+ext, model)
    os.link(mname+ext, mname)
    simplejson.save(mname+".json", model_meta)
    os.unlink(mname+ext, mname)
