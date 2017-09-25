from pylab import *
import numpy as np
import scipy as sc
import scipy.ndimage as ndi
import pylab as pl
import matplotlib as mpl
from IPython import display
from itertools import islice
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
