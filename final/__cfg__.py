import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_

# TODO:
#   - requirements.txt
#   - README.md

gr.update_rcParams("liron_utils-article")
# gr.update_rcParams("liron_utils-text_color", "white")
gr.update_rcParams({
    'figure.autolayout':     False,
    'figure.figsize':        [15, 8],  # figure size in inches
    'figure.dpi':            100,
    # The figure subplot parameters.  All dimensions are a fraction of the figure width and height.
    'figure.subplot.left':   0.05,  # the left side of the subplots of the figure
    'figure.subplot.right':  0.95,  # the right side of the subplots of the figure
    'figure.subplot.bottom': 0.05,  # the bottom of the subplots of the figure
    'figure.subplot.top':    0.93,  # the top of the subplots of the figure
    'figure.subplot.wspace': 0.2,  # the amount of width reserved for space between subplots,
    # expressed as a fraction of the average axis width
    'figure.subplot.hspace': 0.2,  # the amount of height reserved for space between subplots,
    # expressed as a fraction of the average axis height

    'savefig.format':        'tif',
})

# %% Debug
DEBUG = False
SEED = 0  # Use a fixed seed for reproducibility in tests (used only if DEBUG is True)

# %% Paths
PATH_DATA = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data"  # Path to the data directory
get_path = lambda *args: os.path.join(PATH_DATA, *args)

EXCEL_COLUMNS = dict_(
        date="Date",
        pos="Pos",
        time_after_cut="time after cut [min]",
        time_interval="time interval[min]",
        main_orientation="main orientation:\n1-head\n2-foot\n3-side",
        initial_frame_beta_catenin="initial frame of B-catenin",
        final_frame_beta_catenin="final frame of beta catenin",
        beta_catenin_intensity="beta catenin expression intensity:\n0-none;\n1-medium-low\n2-medium-high\n3-max",
)

DIR_OUTPUT = "output"


class DataType:
    def __init__(self, dirname, ext):
        self.dirname = dirname  # Directory name relative to the root directory
        self.ext = ext  # File extension

    def __repr__(self):
        return f"DataType(" + f", ".join(f"{name}={getattr(self, name)}" for name in vars(self).keys()) + ")"


DATA_TYPES = dict_(
        image=DataType(dirname="", ext=".tif"),
        labels=DataType(dirname="labels", ext=".tif"),
        prob=DataType(dirname=os.path.join(DIR_OUTPUT, "random_forest_prob"), ext=".pkl"),
        cpsam_out=DataType(dirname=os.path.join(DIR_OUTPUT, "cpsam_out"), ext=".pkl"),
        figs=DataType(dirname=os.path.join(DIR_OUTPUT, "figs"), ext=f".{plt.rcParams['savefig.format']}"),
)

IGNORED_DIRS = {DIR_OUTPUT, DATA_TYPES.labels.dirname}
"""
└── dir_images
│   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif.lnk
│   ├── ...
│   └── labels
│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif
│   │   ├── ...
│   └── output
│   │   └── random_forest_prob
│   │   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.pkl
│   │   │   ├── ...
│   │   └── cpsam_out
│   │   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.pkl
│   │   │   ├── ...
│   │   └── figs
│   │   │   └── classification
│   │   │   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif
│   │   │   │   ├── ...
│   │   │   ├── classification_movie.gif
│   │   │   ├── stats.tif
"""

# %% Data Manager
CACHE_SIZE = 20  # Maximum number of frames to keep in memory. If exceeded, the oldest image will be removed from memory
AUTO_CONTRAST = True
AUTO_CONTRAST_KW = dict_(
        low_clip_percent=1,
        high_clip_percent=0.015
)

# %% Random Forest Pixel Classifier
SIGMAS = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0)
RANDOM_FOREST_CLASSIFIER_KW = dict_(
        n_estimators=100,
        max_depth=None,
        n_jobs=-2,
)

# TODO:
#   - cellpose.dynamics.compute_masks() -> get_masks_torch()
#   - cellpose.utils.fill_holes_and_remove_small_masks()

# %% Cellpose Model (CPSAM)
CPSAM_EVAL_KW = dict_(
        batch_size=1,  # Unfortunately, batching is done in each image separately, therefore there is no speedup in
        # using batch_size > 1. DataManger is ready for batching, but the model is not.
        diameter=20,
        # flow_threshold=0.2,
        cellprob_threshold=2.0,
        min_size=7 ** 2,  # min nuclei size in pixels
        max_size_fraction=20 ** 2 / 488 ** 2,  # max nuclei size (fraction relative to image size)
)


class CPSAMEvalOut:
    """Container for Cellpose model output."""

    def __init__(self, mask, flow, style):
        self.mask = mask.astype("uint16")
        self.flow = flow
        self.style = style


# %% Ilastik
PATH_ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe"  # Path to the Ilastik's 'run_ilastik.bat' script used for headless processing

# %% Napari
pass

# %% Plots
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

set_props_kw_image = dict(axis="image", ticks=False, xy_lines=False)

LABELS = ["Background", "Nuclei", "Hydra", "Dirt"]
LABELS2IDX = dict_(zip(LABELS, range(1, len(LABELS) + 1)))

CMAP = dict_(
        rgba=ListedColormap(np.array([
            # [R, G, B, alpha]
            [0.471, 0.145, 0.024, 0.00],  # Background
            [0.357, 0.835, 0.973, 0.90],  # Nuclei
            [0.573, 0.537, 0.910, 0.00],  # Hydra
            [0.424, 0.008, 0.757, 0.00],  # Dirt
        ])),
)
CMAP.rgb = ListedColormap(CMAP.rgba.colors[:, :3])
CMAP.rgba_mask = ListedColormap(CMAP.rgba.colors[:2, :])
CMAP.rgb_mask = ListedColormap(CMAP.rgba.colors[:2, :3])


class Stats:
    """Container for nuclei statistics."""

    def __init__(self, count, avg_intensity, avg_area, sum_area_intensity, avg_dist, intensity_excel):
        self.count = count
        self.avg_intensity = avg_intensity
        self.avg_area = avg_area
        self.sum_area_intensity = sum_area_intensity
        self.avg_dist = avg_dist
        self.intensity_excel = intensity_excel


# %% TQDM
TQDM_KW = dict(
        disable=False,
        desc="Processing",
        delay=0.2,
)


def get_tqdm_kw(**tqdm_kw):
    """Return a dictionary of keyword arguments for tqdm."""
    return TQDM_KW | tqdm_kw


# %% Logger
logger = Logger(
        min_level_file=Logger.NAME2LEVEL.DEBUG if DEBUG else Logger.NAME2LEVEL.INFO,
        min_level_console=Logger.NAME2LEVEL.DEBUG if DEBUG else Logger.NAME2LEVEL.INFO,
)

# %% Random seed
if DEBUG:
    np.random.seed(SEED)
    random.seed(SEED)
