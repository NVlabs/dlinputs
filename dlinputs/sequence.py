# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import dbm
import itertools

import numpy as np
from scipy.ndimage import measurements

import utils
import improc


def intlist_to_hotonelist(cs, nc, allow_bad_classes=True):
    """Helper function for LSTM/CTC-based OCR: encode ground truth as array.

    Given a list of target classes `cs` and a total
    maximum number of classes, compute an array that has
    a `1` in each column and time step corresponding to the
    target class, with class 0 interspersed.

    :param cs: list of target classes
    :param nc: total number of classes
    :returns: ndarray representing a hotone encoding

    """
    result = np.zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0] = 1.0
        if allow_bad_classes:
            j = min(j, nc-1) # FIX for bad inputs
        result[2*i+1,j] = 1.0
    result[-1,0] = 1.0
    return result

def hotonelist_to_intlist(outputs,threshold=0.7,pos=0):
    """Helper function for LSTM-based OCR: decode LSTM outputs.

    Translate back. Thresholds on class 0, then assigns the maximum class to
    each region. ``pos`` determines the depth of character information returned:
    - `pos=0`: Return list of recognized characters
    - `pos=1`: Return list of position-character tuples
    - `pos=2`: Return list of character-probability tuples


    :param outputs: 2D array containing posterior probabilities
    :param threshold: posterior probability threshold
    :param pos: what to return
    :returns: decoded hot one outputs

    """
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = np.tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,np.arange(1,np.amax(mask)+1))
    if pos==1: return maxima # include character position
    if pos==2: return [(r, c, outputs[r,c]) for (r,c) in maxima] # include character probabilities
    return [c for (r,c) in maxima] # only recognized characters

def seq_makebatch(images, for_target=False):
    """Given a list of LD sequences, make a BLD batch tensor.

    This performs zero padding as necessary.

    :param images: list of images as LD sequences
    :param for_target: require ndim==2, inserts training blank steps.
    :returns: batched image sequences

    """
    assert isinstance(images, list), type(images)
    assert isinstance(images[0], np.ndarray), images
    if images[0].ndim==2:
        l, d = np.amax(np.array([img.shape for img in images], 'i'), axis=0)
        ibatch = np.zeros([len(images), int(l), int(d)])
        if for_target:
            ibatch[:, :, 0] = 1.0
        for i, image in enumerate(images):
            l, d = image.shape
            ibatch[i, :l, :d] = image
        return ibatch
    elif images[0].ndim==3:
        assert not for_target
        h, w, d = np.amax(np.array([img.shape for img in images], 'i'), axis=0)
        ibatch = np.zeros([len(images), h, w, d])
        for i, image in enumerate(images):
            h, w, d = image.shape
            ibatch[i, :h, :w, :d] = image
        return ibatch

class GenericCodec(object):
    def encode_tensor(self, s):
        codes = [self.encode_char(c) for c in s]
        return intlist_to_hotonelist(codes, self.size())
    def decode_tensor(self, a, threshold=0.7, pos=0):
        codes = hotonelist_to_intlist(a, threshold=threshold, pos=pos)
        if pos==0:
            return "".join([self.decode_char(c) for c in codes])
        elif pos==1:
            return [(r, self.decode_char(c)) for r, c in codes]
        elif pos==2:
            return [(r, self.decode_char(c), p) for r, c, p in codes]
    def encode_batch(self, batch):
        return seq_makebatch([self.encode_tensor(s) for s in batch], for_target=True)
    def decode_batch(self, a, threshold=0.7, pos=0):
        return  [self.decode_tensor(x, threshold=threshold, pos=pos) for x in a]

class AsciiCodec(GenericCodec):
    """An example of a codec, used for turning strings into tensors."""
    def encode_char(self, c):
        if c=="": return 0
        return max(1, ord(c) - ord(" ") + 1)
    def decode_char(self, c):
        if c==0: return ""
        return chr(ord(" ") + c - 1)
    def size(self):
        """The number of classes. Zero is always reserved for the empty class.
        """
        return 97

ascii_codec = AsciiCodec()
