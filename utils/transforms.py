import random
import cv2
import numpy as np
import numbers

__all__ = ['GroupRandomCropRatio', 'GroupRandomScale', 'GroupNormalize']


class GroupRandomCropRatio(object):
    def __init__(self, size, interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        tw, th = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


class GroupRandomScale(object):
    def __init__(self, size, interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        if isinstance(self.size, tuple):
            assert len(self.size) == 2, 'The length of size tuple should be 2.'
            t_w = random.uniform(self.size[0], self.size[1])
        else:
            t_w = self.size
        w = img_group[0].shape[1]
        scale = t_w / w
        out_images = list()
        for img, interpolation in zip(img_group, self.interpolation):
            out_images.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_group):
        out_images = list()
        for img, m, s in zip(img_group, self.mean, self.std):
            if len(m) == 1:
                img = img - np.array(m)  # single channel image
                img = img / np.array(s)
            else:
                img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                img = img / np.array(s)[np.newaxis, np.newaxis, ...]
            out_images.append(img)

        return out_images
