# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import sqlite3

sqltypes = dict(int="integer", str="text", png="blob", float="real")


def identity(x): return x


def saveimg(img):
    return buffer(dli.pildumps(array(img * 255.0, 'uint8')))


converters = dict(png=saveimg)


class DbWriter(object):
    """A quick and simple way of writing datasets to sqlite3 files.

    ```
    dbw = DbWriter("foo.db", "mytable", input="png", transcript="text", cls="int")
    dbw.add(input=rand(128, 128), transcript="foo", cls=17)
    ...
    ```
    """

    def __init__(self, fname, tname, **kw):
        self.db = db = sqlite3.connect(fname)
        self.c = c = db.cursor()
        self.keys = keys = kw.keys()
        decls = []
        cols = []
        vals = []
        self.converters = {}
        for name in keys:
            kind = kw[name]
            decls += [name + " " + sqltypes.get(kind, "text")]
            cols += [name]
            vals += [":" + name]
            self.converters[name] = converters.get(kind, identity)
        decls = ",".join(decls)
        cols = ",".join(cols)
        vals = ",".join(vals)
        c.execute("drop table if exists {tname}".format(tname=tname))
        c.execute("create table {tname} ({decls})".format(
            tname=tname, decls=decls))
        self.insert = "insert into {tname} ({cols}) values ({vals})".format(
            tname=tname, cols=cols, vals=vals)

    def add_dict(self, d):
        d = {k: self.converters[k](v) for k, v in d.items()}
        self.c.execute(self.insert, d)
        self.db.commit()

    def add(self, **kw):
        self.add_dict(kw)
