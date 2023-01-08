from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
from torchvision.transforms import RandomAffine
# from torchvision.transforms.functional import InterpolationMode

def affine(image, shear):
    random_affine = RandomAffine(degrees=0, translate=None, scale=None, shear=shear, resample=Image.BILINEAR)
    return random_affine(image)

def transform_rotate(image, mask):
    angle = transforms.RandomRotation.get_params([0, 36])
    image = image.rotate(angle, expand=True)


    for i in range(len(mask)):
        mask[i] = mask[i].rotate(angle, expand=True)

    # image = tf.to_tensor(image)
    # mask = tf.to_tensor(mask)
    return image, mask

def transform_shear(image, mask):
    angle = transforms.RandomRotation.get_params([-10, 10])
    if random.random() > 0.5:
        image = affine(image, shear=(angle, angle, 0, 0))
        for i in range(len(mask)):
            mask[i] = affine(mask[i], shear=(angle, angle, 0, 0))
    else:
        image = affine(image, shear=(0, 0, angle, angle))
        for i in range(len(mask)):
            mask[i] = affine(mask[i], shear=(0, 0, angle, angle))

    return image, mask

def transform_flip(image, mask):
    if random.random() > 0.5:
        image = tf.hflip(image)
        for i in range(len(mask)):
            mask[i] = tf.hflip(mask[i])
    else:
        image = tf.vflip(image)
        for i in range(len(mask)):
            mask[i] = tf.vflip(mask[i])

    return image, mask


def transform_translate_horizontal(image, mask, scale=0.5):
    w, h = image.size
    mask_pad = []
    if random.random() > 0.5:
        image = tf.crop(image, top=0, left=0, height=h, width=w - w*scale)
        image_pad = tf.pad(image, padding=[0, 0, int(w*scale), 0], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=0, left=0, height=h, width=w - w*scale)
            mask_pad.append(tf.pad(mask[i], padding=[0, 0, int(w*scale), 0], fill=0))

    else:
        image = tf.crop(image, top=0, left=w*scale, height=h, width=w - w * scale)
        image_pad = tf.pad(image, padding=[int(w * scale), 0, 0,  0], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=0, left=w*scale, height=h, width=w - w * scale)
            mask_pad.append(tf.pad(mask[i], padding=[int(w * scale), 0, 0,  0], fill=0))

    return image_pad, mask_pad


def transform_translate_vertical(image, mask, scale=0.5):
    w, h = image.size

    mask_pad = []
    if random.random() > 0.5:
        image = tf.crop(image, top=h*scale, left=0, height=h - h*scale, width=w)
        image_pad = tf.pad(image, padding=[0, int(h*scale), 0,  0], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=h*scale, left=0, height=h - h*scale, width=w)
            mask_pad.append(tf.pad(mask[i], padding=[0, int(h*scale), 0,  0], fill=0))

    else:
        image = tf.crop(image, top=0, left=0, height=h - h*scale, width=w)
        image_pad = tf.pad(image, padding=[0, 0, 0, int(h * scale)], fill=0)

        for i in range(len(mask)):
            mask[i] = tf.crop(mask[i], top=0, left=0, height=h - h*scale, width=w)
            mask_pad.append(tf.pad(mask[i], padding=[0, 0, 0, int(h * scale)], fill=0))

    return image_pad, mask_pad

if __name__ == '__main__':
    img_path = r''
    label_path = r''
    img = Image.open(img_path).convert('L')
    label = Image.open(label_path).convert('L')

    print(img.size)
    image = transform_translate_vertical(img, label)
    plt.imshow(image, cmap='gray')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.show()

