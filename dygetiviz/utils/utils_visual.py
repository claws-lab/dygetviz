import os
import os.path as osp
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import const

def rgb_to_hex(color):
    # Convert a tuple of RGB values to a hex string
    r, g, b = [int(c * 255) for c in color]
    return f'#{r:02x}{g:02x}{b:02x}'

def random_color(hex=False):
    """Archived as the color is not visually appealing

    hex: bool
    """
    if hex:
        color = "#" + ''.join(
            [random.choice('0123456789ABCDEF') for j in range(6)])
    else:
        color = (
            random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    return color


def get_colors(num_colors):
    """Get a list of colors through seaborn

    :param num_colors:
    :return:
    """
    import seaborn as sns

    # Use the hls color space from seaborn128
    colors = sns.color_palette('hls', n_colors=num_colors)

    # Convert the color tuples to hex strings
    colors = [rgb_to_hex(color) for color in colors]

    return colors
