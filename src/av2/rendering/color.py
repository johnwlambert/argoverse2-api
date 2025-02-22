# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Colormap related constants and functions."""

from typing import Final, Sequence, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from av2.utils.typing import NDArrayFloat

RED_HEX: Final[str] = "#df0101"
GREEN_HEX: Final[str] = "#31b404"

RED_RGB: Final[Tuple[int, int, int]] = (255, 0, 0)
RED_BGR: Final[Tuple[int, int, int]] = RED_RGB[::-1]

BLUE_RGB: Final[Tuple[int, int, int]] = (0, 0, 255)
BLUE_BGR: Final[Tuple[int, int, int]] = BLUE_RGB[::-1]

HANDICAP_BLUE_RGB: Final[Tuple[int, int, int]] = (42, 130, 193)
HANDICAP_BLUE_BGR: Final[Tuple[int, int, int]] = HANDICAP_BLUE_RGB[::-1]

WHITE_RGB: Final[Tuple[int, int, int]] = (255, 255, 255)
WHITE_BGR: Final[Tuple[int, int, int]] = WHITE_RGB[::-1]

GRAY_BGR: Final[Tuple[int, int, int]] = (168, 168, 168)
DARK_GRAY_BGR: Final[Tuple[int, int, int]] = (100, 100, 100)

TRAFFIC_YELLOW1_RGB: Final[Tuple[int, int, int]] = (250, 210, 1)
TRAFFIC_YELLOW1_BGR: Final[Tuple[int, int, int]] = TRAFFIC_YELLOW1_RGB[::-1]


def create_colormap(color_list: Sequence[str], n_colors: int) -> NDArrayFloat:
    """Create hex colorscale to interpolate between requested colors.

    Args:
        color_list: list of requested colors, in hex format.
        n_colors: number of colors in the colormap.

    Returns:
        array of shape (n_colors, 3) representing a list of RGB colors in [0,1]
    """
    cmap = LinearSegmentedColormap.from_list(name="dummy_name", colors=color_list)
    colorscale: NDArrayFloat = np.array([cmap(k * 1 / n_colors) for k in range(n_colors)])
    # ignore the 4th alpha channel
    return colorscale[:, :3]  # type: ignore
