#!/usr/bin/env python3

from PIL import Image
import matplotlib.image as mpimg
import os
import numpy as np

# Make tiny palette Image, one black pixel
def convert(path):
    palIm = Image.new('P', (1,1))

    palette = [ 
        255, 195, 195,
        255, 161, 255,
        92, 1, 0,
        210, 0, 0,
        255, 170, 1,
        255, 95, 1,
        255, 79, 190,
        170, 0, 255,
        255, 255, 0,
        0, 43, 0,
        0, 124, 0,
        124, 255, 123,
        24, 0, 62,
        0, 0, 255,
        0, 144, 255,
        0, 255, 169,
        123, 198, 255,
        0, 255, 255,
        186, 254, 255,
        255, 255, 255,
        202, 202, 202,
        129, 129, 129,
        65, 65, 65,
        0, 0, 0,
        83, 30, 0,
        2, 1, 1,
        61, 0, 93,
        255, 196, 78,
        255, 223, 162,
        122, 100, 53,
        38, 32, 18,
        175, 103, 62
    ]


    # Push in our lovely B&W palette and save just for debug purposes
    palIm.putpalette(palette)

    # Load actual image 
    actual = Image.open(path).convert('RGB')

    # Quantize actual image to palette
    res = np.array(actual)
    print(res)
    actual = actual.quantize(palette=palIm, dither=Image.Dither.NONE)
    # actual = actual.quantize(colors=16)

    # actual.save(f'{path}_converted.png')

if __name__ == '__main__':
    convert('images\woman.jpg')