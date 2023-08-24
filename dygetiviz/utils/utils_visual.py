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


def get_colors(num_colors, palette_name="Spectral"):
    """Get a list of colors through seaborn

    :param palette_name: Name of the color palette, such as `hls`, `Spectral`
    :param num_colors:
    :return:
    """
    import seaborn as sns

    # Use the hls color space from seaborn128
    colors = sns.color_palette(palette_name, n_colors=num_colors)

    # Convert the color tuples to hex strings
    colors = [rgb_to_hex(color) for color in colors]

    return colors


def get_hovertemplate(fields_in_customdata, is_trajectory=False):

    if is_trajectory:
        hovertemplate = '<b>%{hovertext}</b><br><br>node_color=#B2B2B2<br>x=%{x}<br>y=%{y}<br>display_name=%{text}<br><br>'

    else:
        hovertemplate = '<b>%{hovertext}</b><br><br>node_color=#B2B2B2<br>x=%{x}<br>y=%{y}<br><br>'

    for i, field in enumerate(fields_in_customdata):
        # hovertemplate += f"{field}=" + "%{" + f"hover_data_{i}" + "}<br>"
        hovertemplate += f"{field}=" + "%{" + f"customdata[{i}]" + "}<br>"

    hovertemplate += "<extra></extra>"

    return hovertemplate