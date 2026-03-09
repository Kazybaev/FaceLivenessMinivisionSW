"""Custom image transforms.

CRITICAL: to_tensor() does NOT divide by 255. This is load-bearing for model weights.
Original comment: "return img.float().div(255)  modify by zkx" → img.float()

Original: src/data_io/functional.py + src/data_io/transform.py (Minivision, zhuying)
"""
from __future__ import annotations

import math
import random
import numbers

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

try:
    import accimage
except ImportError:
    accimage = None


# ─── Type checks ─────────────────────────────────────────────────────────────

def _is_pil_image(img: object) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    return isinstance(img, Image.Image)


def _is_numpy_image(img: object) -> bool:
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# ─── Core functional transforms ─────────────────────────────────────────────

def to_tensor(pic: np.ndarray | Image.Image) -> torch.Tensor:
    """Convert PIL Image or numpy.ndarray to tensor.

    CRITICAL: Does NOT divide by 255. Model weights depend on this behavior.
    """
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(f'pic should be PIL Image or ndarray. Got {type(pic)}')

    if isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            pic = pic.reshape((pic.shape[0], pic.shape[1], 1))
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # CRITICAL: img.float() without .div(255) — model weights depend on this
        return img.float()

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # Handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        # CRITICAL: img.float() without .div(255)
        return img.float()
    else:
        return img


def to_pil_image(pic: torch.Tensor | np.ndarray, mode: str | None = None) -> Image.Image:
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
    else:
        npimg = pic

    if not isinstance(npimg, np.ndarray):
        raise TypeError(f'Input pic must be a torch.Tensor or NumPy ndarray, not {type(npimg)}')

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) for input type {npimg.dtype}. Expected {expected_mode}")
        mode = expected_mode
    elif npimg.shape[2] == 4:
        permitted = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted:
            raise ValueError(f"Only modes {permitted} are supported for 4D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted:
            raise ValueError(f"Only modes {permitted} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError(f'Input type {npimg.dtype} is not supported')

    return Image.fromarray(npimg, mode=mode)


def resize(img: Image.Image, size: int | tuple, interpolation: int = Image.BILINEAR) -> Image.Image:
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img: Image.Image, i: int, j: int, h: int, w: int) -> Image.Image:
    return img.crop((j, i, j + w, i + h))


def center_crop(img: Image.Image, output_size: int | tuple) -> Image.Image:
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(img, i, j, th, tw)


def resized_crop(
    img: Image.Image, i: int, j: int, h: int, w: int,
    size: int | tuple, interpolation: int = Image.BILINEAR,
) -> Image.Image:
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def rotate(img: Image.Image, angle: float, resample: bool = False,
           expand: bool = False, center: tuple | None = None) -> Image.Image:
    return img.rotate(angle, resample, expand, center)


def adjust_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(brightness_factor)


def adjust_contrast(img: Image.Image, contrast_factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(contrast_factor)


def adjust_saturation(img: Image.Image, saturation_factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(saturation_factor)


def adjust_hue(img: Image.Image, hue_factor: float) -> Image.Image:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].')
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img
    h, s, v = img.convert('HSV').split()
    np_h = np.array(h, dtype=np.uint8)
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')
    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


# ─── Transform classes ──────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: object) -> object:
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    """Convert PIL Image or numpy.ndarray to tensor.

    CRITICAL: Does NOT divide by 255.
    """
    def __call__(self, pic: np.ndarray | Image.Image) -> torch.Tensor:
        return to_tensor(pic)


class ToPILImage:
    def __init__(self, mode: str | None = None):
        self.mode = mode

    def __call__(self, pic: torch.Tensor | np.ndarray) -> Image.Image:
        return to_pil_image(pic, self.mode)


class Lambda:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, img: object) -> object:
        return self.lambd(img)


class RandomHorizontalFlip:
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            return hflip(img)
        return img


class RandomResizedCrop:
    def __init__(
        self,
        size: int | tuple,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: int = Image.BILINEAR,
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Image.Image, scale: tuple, ratio: tuple) -> tuple[int, int, int, int]:
        for _ in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w
        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img: Image.Image) -> Image.Image:
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness: float, contrast: float, saturation: float, hue: float) -> Compose:
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img, bf=brightness_factor: adjust_brightness(img, bf)))
        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img, cf=contrast_factor: adjust_contrast(img, cf)))
        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img, sf=saturation_factor: adjust_saturation(img, sf)))
        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img, hf=hue_factor: adjust_hue(img, hf)))
        np.random.shuffle(transforms)
        return Compose(transforms)

    def __call__(self, img: Image.Image) -> Image.Image:
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return transform(img)


class RandomRotation:
    def __init__(
        self,
        degrees: float | tuple,
        resample: bool = False,
        expand: bool = False,
        center: tuple | None = None,
    ):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        return rotate(img, angle, self.resample, self.expand, self.center)
