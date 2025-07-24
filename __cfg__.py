import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_

# TODO:
#   - README.md

gr.update_rcParams("liron_utils-article")
# gr.update_rcParams("liron_utils-text_color", "white")
gr.update_rcParams({
    'figure.autolayout': False,  # When True, automatically adjust subplot
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
DEBUG = False  # Controls logger level and random seed (debugging can still be achieved with the `--debug` flag)
SEED = 0  # Use a fixed seed for reproducibility in tests (used only if DEBUG is True or with `--debug` flag)

# %% Paths
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
    """Container for data type information."""

    def __init__(self, dirname: str, ext: str, filename: str = None, return_to_user: bool = True):
        self.dirname = dirname  # Directory name relative to the root directory

        if not ext.startswith("."):
            ext = "." + ext
        self.ext = ext  # File extension

        if filename is not None:
            self.filename = filename

        self.return_to_user = return_to_user

    def __repr__(self):
        return f"DataType(" + f", ".join(f"{name}={getattr(self, name)}" for name in vars(self).keys()) + ")"


DATA_TYPES: dict[str, DataType] = dict_(
        # If defining new DataType with `return_to_user=True`, add it to `DataManager.get_data()` and `IGNORED_DIRS`
        image=DataType(
                dirname="",
                ext="tif"
        ),
        labels=DataType(
                dirname="labels",
                ext="tif"
        ),
        prob=DataType(
                dirname=os.path.join(DIR_OUTPUT, "random_forest_prob"),
                ext="pkl"
        ),
        cpsam_out=DataType(
                dirname=os.path.join(DIR_OUTPUT, "cpsam_out"),
                ext="pkl"
        ),
        stats=DataType(
                dirname=os.path.join(DIR_OUTPUT),
                filename="stats",
                ext="pkl",
                return_to_user=False
        ),
        pixel_classifier=DataType(
                dirname=os.path.join(DIR_OUTPUT),
                filename="pixel_classifier",
                ext="pkl",
                return_to_user=False
        ),
        figs=DataType(
                dirname=os.path.join(DIR_OUTPUT, "figs"),
                ext=plt.rcParams['savefig.format'],
                return_to_user=False
        ),
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
class SegmentationLabel:
    """Container for labels and their properties."""

    def __init__(self, idx_napari, color, alpha=1.0):
        self.idx_napari = idx_napari
        self.color = gr.hex2rgb(color)  # Color of the label in RGB format
        self.alpha = alpha  # Transparency of the label (0.0 - fully transparent, 1.0 - fully opaque)

    def __repr__(self):
        return f"SegmentationLabel(idx_napari={self.idx_napari}, color={self.color}, alpha={self.alpha})"


LABELS = dict_(
        background=SegmentationLabel(idx_napari=1, color=gr.COLORS.LIGHT_BROWN, alpha=0.00),
        nuclei=SegmentationLabel(idx_napari=2, color=gr.COLORS.BLUE, alpha=0.90),
        hydra=SegmentationLabel(idx_napari=3, color=gr.COLORS.PURPLE_A, alpha=0.00),
        dirt=SegmentationLabel(idx_napari=4, color=gr.COLORS.PURPLE, alpha=0.00),
)

# %% Plots
IMAGE_EXTENSIONS = {'.tif', '.tiff'}  # '.jpg', '.jpeg', '.png', '.bmp', '.gif'
# If adding new extension, update `utils.imread()` and `utils.imwrite()`

set_props_kw_image = dict(axis="image", ticks=False, xy_lines=False)


def get_cmap(labels=None, alpha=False):
    """Return a colormap for the given labels."""
    if labels is None:
        labels = list(LABELS.values())
    if alpha:
        return ListedColormap(np.vstack([np.hstack([label.color, [label.alpha]]) for label in labels]))
    else:
        return ListedColormap(np.vstack([label.color for label in labels]))


CMAP = dict_(
        rgb=get_cmap(),  # 1=background, 2=nuclei, 3=hydra, 4=dirt
        rgba=get_cmap(alpha=True),  # 1=background, 2=nuclei, 3=hydra, 4=dirt
        rgb_cpsam_mask=get_cmap(labels=[LABELS.background, LABELS.nuclei]),  # 0=background, 1=nuclei
        rgba_cpsam_mask=get_cmap(labels=[LABELS.background, LABELS.nuclei], alpha=True),  # 0=background, 1=nuclei
)


class Stats:
    """Container for nuclei statistics."""

    def __init__(self, size=None,
            count=None, avg_intensity=None, avg_area=None,
            sum_area_intensity=None, avg_dist=None, intensity_excel=None):
        if size is None:
            self.count = count  # number of detected nuclei
            self.avg_intensity = avg_intensity  # average intensity
            self.avg_area = avg_area  # average nuclei area
            self.sum_area_intensity = sum_area_intensity  # sum of area * intensity
            self.avg_dist = avg_dist  # average distance between nuclei
            self.intensity_excel = intensity_excel  # beta-catenin intensity from Excel
            # todo:
            #   - min distance between nuclei
        else:  # initialize with a specific size
            self.count = np.zeros(size, dtype="uint16")  # number of detected nuclei
            self.avg_intensity = np.full(size, np.nan)  # average intensity
            self.avg_area = np.full(size, np.nan)  # average nuclei area
            self.sum_area_intensity = np.full(size, np.nan)  # sum of area * intensity
            self.avg_dist = np.full(size, np.nan)  # average distance between nuclei
            self.intensity_excel = None  # beta-catenin intensity from Excel


# %% TQDM
def get_tqdm_kw(**tqdm_kw):
    """Return a dictionary of keyword arguments for tqdm."""
    return dict(
            disable=False,
            desc="Processing",
            delay=0.2,
    ) | tqdm_kw


# %% Logger
logger = Logger(
        min_level_file=Logger.NAME2LEVEL.DEBUG if DEBUG else Logger.NAME2LEVEL.INFO,
        min_level_console=Logger.NAME2LEVEL.DEBUG if DEBUG else Logger.NAME2LEVEL.INFO,
)
