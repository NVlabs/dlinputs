# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import inspect


class LocalImport(object):

  def __init__(self, names):
    if not isinstance(names, dict):
      names = vars(names)
    self.names = names

  def __enter__(self):
    self.frame = inspect.currentframe()
    bindings = self.frame.f_back.f_globals
    self.old_bindings = {k: bindings.get(k, None) for k in self.names.keys()}
    bindings.update(self.names)

  def __exit__(self, some_type, value, traceback):
    del some_type, value, traceback
    bindings = self.frame.f_back.f_globals
    bindings.update(self.old_bindings)
    extras = [k for k, v in self.old_bindings.items() if v is None]
    for k in extras:
        del bindings[k]
    del self.frame
