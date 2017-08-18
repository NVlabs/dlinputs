#!/usr/bin/python

import dlinputs as dli

class Inputs(object):
    def training_data(self, **kw):
        return dli.itsqlite("testdata/sample.db", **kw) | \
               dli.itmap(image=dli.pilreads, cls=int)
