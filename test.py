from itertools import islice

import numpy as np
import pylab as pl
import scipy as sc
import matplotlib as mpl
import scipy.ndimage as ndi
from pylab import *
from IPython import display

import dlinputs; reload(dlinputs); dli = dlinputs

urlpath = """
http://ixion:9000/
http://sedna:9000/
http://localhost:9000/
"""

data = (dli.ittarshards("imagenet.shards", urlpath=urlpath) |
        dli.itmap(png=dli.pilrgb, cls=int))
for sample in data:
    print sample["cls"]
    imshow(sample["png"])
    show()
    break
