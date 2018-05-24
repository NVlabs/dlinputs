from __future__ import unicode_literals
from builtins import str
from builtins import range
import dlinputs.filters as dlf
from imp import reload
reload(dlf)

import numpy as np

def test_compose():
  def f(it):
      for x in it: yield x+1
  g = dlf.compose(f, f, f)
  assert list(g(range(5))) == [3,4,5,6,7]

  reload(dlf)
  def f(s):
      def g(it):
          for x in it: yield x+s
      return g
  g = dlf.compose(f("a"), f("b"), f("c"), f("d"))
  assert list(g([""])) == ["abcd"]

def source(n=5):
    for i in range(n):
        yield dict(__key__="{:06d}".format(i))

def source2(n=5):
    for i in range(n):
        yield dict(__key__="{:06d}".format(i),
                   png=np.zeros((1, 1, 3)),
                   cls=i%3,
                   info=str(i))

def test_source():
  assert len(list(dlf.merge([source()]))) == 5
  assert len(list(dlf.merge([source(), source()]))) == 10

def test_concat():
  assert len(list(dlf.concat([source()]))) == 5
  assert len(list(dlf.concat([source(), source()]))) == 10

def test_info():
  list(dlf.info(every=2)(source()));

def test_grep():
  assert list(dlf.grep(info="2")(source2()))[0]["cls"] == 2

def test_select():
  result = list(dlf.select(cls=lambda x: x<2)(source2()))
  assert len(result)==4, result

def test_rename():
  results = list(dlf.rename(klasse="cls")(source2()))
  keys = list(set(tuple(sorted(x.keys())) for x in results))[0]
  assert "cls" not in keys
  assert "klasse" in keys

def test_copy():
  results = list(dlf.copy(klasse="cls")(source2()))
  keys = list(set(tuple(sorted(x.keys())) for x in results))[0]
  assert "cls" in keys
  assert "klasse" in keys

def test_map():
  results = list(dlf.map(cls=lambda x: 99)(source2()))
  classes = set(x["cls"] for x in results)
  assert classes == set([99])

def test_transform():
  results = list(dlf.transform(lambda x: dict(q=x["cls"]))(source2()))
  assert results[0]["q"] == 0
  assert set(results[0].keys()) == set("q __key__".split())

def test_shuffle():
  initial = list(source(200))
  results = list(dlf.shuffle(1000, 100)(x for x in initial))
  assert len(results) == 200
  before = set(x["__key__"] for x in initial)
  after = set(x["__key__" ] for x in results)
  assert len(before) == 200
  assert before==after

def test_batched():
  for sample in dlf.batched(20)(source2(100)):
      assert sample["png"].shape == (20, 1, 1, 3)

def test_compose():
  for sample in dlf.compose(dlf.batched(20), dlf.unbatch())(source2(100)):
      assert sample["png"].shape == (1, 1, 3), sample["png"].shape
