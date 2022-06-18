import numpy as np
import random
from PIL import Image, ImageEnhance


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        bright_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        bright_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        sat_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        sat_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return bright_factor, contrast_factor, sat_factor, hue_factor


def apply_jitter(img, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: _adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: _adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(lambda img: _adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: _adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    jittered_img = apply_jitter(
        img,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    return jittered_img


def transform_img(img, affine_trans, res):

    rev_affine_trans = np.linalg.inv(affine_trans)
    img = img.transform(
        tuple(res),
        Image.AFFINE,
        (
            rev_affine_trans[0, 0],
            rev_affine_trans[0, 1],
            rev_affine_trans[0, 2],
            rev_affine_trans[1, 0],
            rev_affine_trans[1, 1],
            rev_affine_trans[1, 2],
        ),
    )
    return img


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def _adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def _adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def _adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor {} is not in [-0.5, 0.5].".format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")

    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img
