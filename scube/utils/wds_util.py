import re
import io
import json
import boto3
import numpy as np

from webdataset.handlers import reraise_exception
from webdataset import gopen

from webdataset.tariterators import (
    base_plus_ext,
    valid_sample,
)
from webdataset.autodecode import IMAGE_EXTENSIONS

imagespecs = {
    "npraw": ("numpy", None, None), # add npraw for decoding uint16 type
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    "torch": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}

class ImageHandler:
    """Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - npraw: numpy None None
    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba

    """

    def __init__(self, imagespec, extensions=IMAGE_EXTENSIONS):
        """Create an image handler.

        :param imagespec: short string indicating the type of decoding
        :param extensions: list of extensions the image handler is invoked for
        """
        if imagespec not in list(imagespecs.keys()):
            raise ValueError("Unknown imagespec: %s. \n\
                              If it is `npraw` (numpy raw), you shoud pip install git+https://github.com/yifanlu0227/webdataset.git rather than the official one" % imagespec)
        self.imagespec = imagespec.lower()
        self.extensions = extensions

    def __call__(self, key, data):
        """Perform image decoding.

        :param key: file name extension
        :param data: binary data
        """
        import PIL.Image

        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            if mode is not None:
                img = img.convert(mode.upper())

        if atype == "pil":
            if mode == "l":
                img = img.convert("L")
                return img
            elif mode == "rgb":
                img = img.convert("RGB")
                return img
            elif mode == "rgba":
                img = img.convert("RGBA")
                return img
            else:
                raise ValueError("Unknown mode: %s" % mode)

        result = np.asarray(img)

        if etype == "float":
            result = result.astype(np.float32) / 255.0

        assert result.ndim in [2, 3], result.shape
        assert mode in ["l", "rgb", "rgba", None], mode

        if mode == "l":
            if result.ndim == 3:
                result = np.mean(result[:, :, :3], axis=2)
        elif mode == "rgb":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
            elif result.shape[2] == 4:
                result = result[:, :, :3]
        elif mode == "rgba":
            if result.ndim == 2:
                result = np.repeat(result[:, :, np.newaxis], 4, axis=2)
                result[:, :, 3] = 255
            elif result.shape[2] == 3:
                result = np.concatenate(
                    [result, 255 * np.ones(result.shape[:2])], axis=2
                )

        assert atype in ["numpy", "torch"], atype

        if atype == "numpy":
            return result
        elif atype == "torch":
            import torch

            if result.ndim == 3:
                return torch.from_numpy(result.transpose(2, 0, 1).copy())
            else:
                return torch.from_numpy(result.copy())

        return None


def imagehandler(imagespec, extensions=IMAGE_EXTENSIONS):
    """Create an image handler.

    This is just a lower case alias for ImageHander.

    :param imagespec: textual image spec
    :param extensions: list of extensions the handler should be applied for
    """
    return ImageHandler(imagespec, extensions)


def get_bytes_io(path, client):
    _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", path)
    for attempt in range(10):
        try:
            byte_io = io.BytesIO()
            client.download_fileobj(bucket, key, byte_io)
            byte_io.seek(0)
            break
        except:
            import os

            if "NODE_RANK" in os.environ:
                print(
                    f"fail to load {path}, attempt: {attempt}, in node, ",
                    os.environ["NODE_RANK"],
                )
            else:
                print(f"fail to load {path}, attempt: {attempt}")
    return byte_io


def s3_url_opener(
    data,
    handler = reraise_exception,
    config_file = 's3/pbss_credentials_default.secret',
    s3_cfg = None,
    **kw,
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """

    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        if url.startswith("s3"):
            if s3_cfg is None:
                with open(config_file, "r") as f:
                    s3_cfg = json.load(f)
            s3_cfg["region_name"] = "us-east-1"
            client = boto3.client("s3", **s3_cfg)
            stream = get_bytes_io(url, client)
            sample.update(stream=stream)
            yield sample
        else:
            try:
                stream = gopen.gopen(url, **kw)
                sample.update(stream=stream)
                yield sample
            except Exception as exn:
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                else:
                    break


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def file_selector(sample, keys):
    for k in keys:
        if k in sample:
            return True
    return False


def base_plus_ext_custom(path, video_frames=False):
    '''
    if vid_frames:
        xxxx/00.01.rgb.png -> xxxx/00, 01.rgb.png
        xxxx/00.meta.json 
    else:
        xxxx/00.01.rgb.png -> xxxx/00.01, rgb.png
        xxxx/00.meta.json -> xxxx/00, meta.json
    '''
    if video_frames:
        match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    else:
        match = re.match(r"^((?:.*/|)[^.]+.[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)