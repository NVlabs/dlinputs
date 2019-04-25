from __future__ import division, print_function

import collections
import functools as ft
import io
import re
import sys
from builtins import range

import numpy as np
import PIL
import PIL.Image
import six
from future import standard_library
from past.utils import old_div

# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

standard_library.install_aliases()


if sys.version_info[0] == 3:
    from builtins import str
    unicode = str
    buffer = str


def print_sample(sample):
    """Pretty print a standard sample.

    :param dict sample: key value pairs used for training

    """
    for k in sorted(sample.keys()):
        v = sample[k]
        print(k, end=' ')
        if isinstance(v, np.ndarray):
            print(v.dtype, v.shape)
        # Unicode change, very simple. If str or unicode, print as is
        elif isinstance(v, six.string_types):
            print(repr(v)[:60])
        elif isinstance(v, (int, float)):
            print(v)
        elif isinstance(v, buffer):
            print(type(v), len(v))
        else:
            print(type(v), repr(v)[:60])


def type_info(x, use_size=True):
    if isinstance(x, np.ndarray):
        if use_size:
            return (x.dtype,) + tuple(x.shape)
        else:
            return (x.dtype, tuple([True]*len(x.shape)))
    else:
        return repr(type(x))


def summarize_samples(source, use_size=True):
    counter = collections.Counter()
    for sample in source:
        descriptor = [(k, type_info(v, use_size))
                      for k, v in list(sample.items())]
        descriptor = tuple(sorted(descriptor))
        counter.update([descriptor])
    return counter


def make_gray(image):
    """Converts any image to a grayscale image by averaging.

    Knows about alpha channels etc.

    :param image: rank 2 or 3 ndarray
    :returns: rank 2 ndarray

    """
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim == 2:
        return image
    assert image.ndim == 3
    assert image.shape[2] in [1, 3, 4]
    return np.mean(image[:, :, :3], 2)


def make_rgb(image):
    """Converts any image to an RGB image.

    Knows about alpha channels etc.

    :param image: rank 2 or 3 ndarray
    :returns: rank 3 ndarray of shape :,:,3

    """
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim == 2:
        image = image.reshape(image.shape + (1,))
    assert image.ndim == 3
    if image.shape[2] == 1:
        return np.repeat(image, 3, 2)
    elif image.shape[2] == 3:
        return image
    elif image.shape[2] == 4:
        return image[:, :, :3]


def make_rgba(image, alpha=255):
    """Converts any image to an RGBA image.

    Knows about alpha channels etc.

    :param image: rank 2 or 3 ndarray
    :param alpha: default alpha value
    :returns: rank 3 ndarray with shape :,:,4

    """
    assert isinstance(image, np.ndarray), type(image)
    if image.ndim == 2:
        image = image.reshape(image.shape + (1,))
    assert image.ndim == 3
    if image.shape[2] == 1:
        result = np.repeat(image, 4, 2)
        result[:, :, 3] = alpha
        return result
    elif image.shape[2] == 3:
        h, w, _ = image.shape
        result = np.zeros((h, w, 4), 'uint8')
        result[:, :, :3] = image
        result[:, :, 3] = alpha
        return result
    elif image.shape[2] == 4:
        return image


def invert_mapping(kvp):
    """Inverts the mapping given by a dictionary.

    :param kvp: mapping to be inverted
    :returns: inverted mapping
    :rtype: dictionary

    """
    return {v: k for k, v in list(kvp.items())}


def get_string_mapping(kvp):
    """Returns a dictionary mapping strings to strings.

    This can take either a string of the form "name=value:name2=value2"
    or a dictionary containing all string keys and values.

    :param kvp: dictionary or string
    :returns: dictionary

    """
    if kvp is None:
        return {}
    # str, unicode change. Very simple. If string type, split and retrun key, value pair
    if isinstance(kvp, six.string_types):
        return {k: v for k, v in [kv.split("=", 1) for kv in kvp.split(":")]}
    elif isinstance(kvp, dict):
        for k, v in list(kvp.items()):
            assert isinstance(k, str)
            assert isinstance(v, str)
        return kvp
    else:
        raise ValueError("{}: wrong type".format(type(kvp)))


def pilread(stream, color="gray", asfloat=True):
    """Read an image from a stream using PIL.

    :param stream: stream to read the image from
    :param color: "gray", "rgb" or "rgba".
    :param asfloat: return float image instead of uint8 image

    """
    if color[-1] == "8":
        asfloat = False
        color = color[:-1]
    image = PIL.Image.open(stream)
    result = np.asarray(image)
    assert result.dtype == np.uint8, image
    if color is None:
        pass
    elif color == "gray":
        result = make_gray(result)
    elif color == "rgb":
        result = make_rgb(result)
    elif color == "rgba":
        result = make_rgba(result)
    else:
        raise ValueError("{}: unknown color space".format(color))
    if asfloat:
        result = result.astype("f") / 255.0
    return result


def pilreads(data, color, asfloat=True):
    """Read an image from a string or buffer using PIL.

    :param data: data to be decoded
    :param color: "gray", "rgb" or "rgba".
    :param asfloat: return float instead of uint8

    """
    assert color is not None
    return pilread(six.BytesIO(data), color=color, asfloat=asfloat)


pilgray = ft.partial(pilreads, color="gray")
pilrgb = ft.partial(pilreads, color="rgb")


def pildumps(image, format="PNG"):
    """Compress an image using PIL and return it as a string.

    Can handle float or uint8 images.

    :param image: ndarray representing an image
    :param format: compression format ("PNG" or "JPEG")

    """
    # BytesIO change very simple. You are creating an image, saving it as bytes to resut which you'll
    # write to disk as Bytes.
    if image.dtype in [np.dtype('f'), np.dtype('d')]:
        assert np.amin(image) > -0.001 and np.amax(image) < 1.001
        image = np.clip(image, 0.0, 1.0)
        image = np.array(image * 255.0, 'uint8')
    with six.BytesIO() as result:
        PIL.Image.fromarray(image).save(result, format=format)
        return result.getvalue()


pilpng = pildumps
piljpg = ft.partial(pildumps, format="JPEG")
format_table = {"png": "PNG-PIL", "jpg": "JPEG-PIL", "jpeg": "JPEG-PIL"}


def autodecode1(data, tname, imagetype="rgb"):
    # Unicode change. If it is alread an unicode string, no decoding (Byte->Unicode req)
    if isinstance(data, (int, float, unicode)):
        return data
    if sys.version_info[0] == 2:
        # Then, it has to be byte string, which is also of type str
        assert isinstance(data, (str, buffer)), type(data)
    else:
        # In Python 3, it has to be a bytes string at this point. You've checked if it is normal string above (unicode check)
        assert isinstance(data, bytes), type(data)
    assert isinstance(tname, str), tname
    extension = re.sub(r".*\.", "", tname).lower()
    if extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        try:
            return int(data)
        except ValueError:
            return data
    elif extension in ["png", "jpg", "jpeg", "img", "image", "pbm", "pgm", "ppm"]:
        if imagetype == "PIL":
            with six.BytesIO(data) as stream:
                img = PIL.Image.open(stream)
                img.load()
                return img
        else:
            return pilreads(data, color=imagetype)
    elif extension in ["json", "jsn"]:
        import simplejson
        return simplejson.loads(data)
    elif extension in ["pyd", "pickle"]:
        import pickle
        return pickle.loads(data)
    elif extension in ["mp", "msgpack", "msg"]:
        import msgpack
        return msgpack.unpackb(data)
    elif extension in ["cls", "cls2", "index", "inx"]:
        return int(str(data))
    return data


def autodecode(sample, imagetype="rgb"):
    """Autodecode a sample, using extensions as guide for how to decode.

    :param sample: dictionary representing sample
    :param imagetype: format for images (gray, rgb, rgba, PIL; rgb8=8 bit)
    """
    result = {}
    for k, v in list(sample.items()):
        if k[0] == "_":
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            result[k] = v
            continue
        assert v is not None, (k, sample)
        result[k] = autodecode1(v, k, imagetype=imagetype)
    return result


def autodecoder(imagetype):
    return lambda sample: autodecode(sample, imagetype=imagetype)


def bytestr(data):
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("ascii")
    return str(data).encode("ascii")


def autoencode1(data, tname):
    extension = re.sub(r".*\.", "", tname).lower()
    if isinstance(data, (int, float)):
        return bytestr(data)
    elif extension in ["png", "jpg", "jpeg", "img", "image", "pbm", "pgm", "ppm"]:
        return pildumps(data, extension)
    if extension in ["json", "jsn"]:
        import simplejson
        return bytestr(simplejson.dumps(data))
    if extension in ["pyd", "pickle"]:
        import pickle
        return pickle.dumps(data)
    if extension in ["mp", "msgpack", "msg"]:
        import msgpack
        return msgpack.packb(data)
    if False and sys.version_info[0] == 3:
        if isinstance(data, str):
            data = data.encode("utf-8")
    return data


def autoencode(sample):
    return {k: autoencode1(v, k) for k, v in list(sample.items())}


def samples_to_batch(samples, combine_tensors=True, combine_scalars=True, expand=False):
    """Take a collection of samples (dictionaries) and create a batch.

    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.

    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict

    """
    if expand:
        return samples_to_batch_expanded(samples)
    result = {k: [] for k in list(samples[0].keys())}
    for i in range(len(samples)):
        for k in list(result.keys()):
            result[k].append(samples[i][k])
    if combine_tensors == True:
        tensor_names = [x for x in list(result.keys())
                        if isinstance(result[x][0], np.ndarray)]
        for k in tensor_names:
            sizes = {a.shape for a in result[k]}
            assert len(sizes) == 1, sizes
            result[k] = np.array(result[k])
    if combine_scalars == True:
        scalar_names = [x for x in list(result.keys())
                        if isinstance(result[x][0], (int, float))]
        for k in scalar_names:
            result[k] = np.array(result[k])
    return result


def samples_to_batch_expanded(samples):
    """Take a collection of samples (dictionaries) and create a batch.

    :param dict samples: list of samples
    :returns: single sample consisting of a batch
    :rtype: dict

    """
    result = {k: [] for k in list(samples[0].keys())}
    for i in range(len(samples)):
        for k in list(result.keys()):
            result[k].append(samples[i][k])
    tensor_names = [x for x in list(result.keys())
                    if isinstance(result[x][0], np.ndarray)]
    for k in tensor_names:
        size = result[k][0].shape
        for r in result[k][1:]:
            size = tuple(np.maximum(size, r.shape))
        output = np.zeros((len(result[k]),) + size)
        for i, t in enumerate(result[k]):
            sub = tuple([i] + [slice(0, x) for x in t.shape])
            output[sub] = t
        result[k] = output
    return result


def metadict(sample, data={}):
    result = {k: v for k, v in list(sample.items()) if k[0] == "_"}
    result.update(data)
    return result
