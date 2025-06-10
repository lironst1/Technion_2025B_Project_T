import os
import numpy as np
from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_, ispc

DEBUG = False

gr.update_rcParams("liron_utils-article")
gr.update_rcParams({
	'figure.autolayout':     False,
	'figure.figsize': [15, 8],  # figure size in inches
	'figure.dpi':            100,
	# The figure subplot parameters.  All dimensions are a fraction of the figure width and height.
	'figure.subplot.left':   0.05,  # the left side of the subplots of the figure
	'figure.subplot.right':  0.95,  # the right side of the subplots of the figure
	'figure.subplot.bottom': 0.05,  # the bottom of the subplots of the figure
	'figure.subplot.top':    0.93,  # the top of the subplots of the figure
	'figure.subplot.wspace': 0.15,  # the amount of width reserved for space between subplots,
	# expressed as a fraction of the average axis width
	'figure.subplot.hspace': 0.15,  # the amount of height reserved for space between subplots,
	# expressed as a fraction of the average axis height

	'savefig.format':        'tif',
})

# %% Constants
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']
PATH_DATA = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data" if ispc \
	else "/Users/lironst/Library/CloudStorage/OneDrive-Technion/Homework/2025B/114252 - Project T/Data"  # Path to the data directory
get_path = lambda *args: os.path.join(PATH_DATA, *args)

DATA_TYPES = dict_(
		image=dict_(dirname="data", ext=".tif"),
		labels=dict_(dirname="labels", ext=".tif"),
		prob=dict_(dirname="prob", ext=".tif"),
		model_out=dict_(dirname="model_out", ext=".pkl"),
)  # {<data_type>: <subdir_name>}

# %% Data Manager
AUTO_CONTRAST = True
# EQUALIZE_ADAPTHIST_KW = dict(clip_limit=0.025)

# %% Pixel Classifier
SIGMAS = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0]
RANDOM_FOREST_CLASSIFIER_KW = dict_(
		n_estimators=100,
		max_depth=None,
		n_jobs=-2,
)

# %% Cellpose


# %% Ilastik
PATH_ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe"  # Path to the Ilastik's 'run_ilastik.bat' script used for headless processing

# %% Napari

# %% Plots
set_props_kw_image = dict(axis="image", ticks=False, xy_lines=False)
LABELS = ["Background", "Nuclei", "Dirt", "Inner Dirt"]
LABELS2IDX = dict_(zip(LABELS, range(1, len(LABELS) + 1)))
CMAP = dict_(
		rgba=ListedColormap(np.array([
			# [R, G, B, alpha]
			[0.471, 0.145, 0.024, 0.00],  # background
			[0.357, 0.835, 0.973, 0.75],  # nuclei
			[0.573, 0.537, 0.910, 0.00],  # dirt
			[0.424, 0.008, 0.757, 0.00],  # inner dirt
		])),
)
CMAP.rgb = ListedColormap(CMAP.rgba.colors[:, :3])
CMAP.rgba_mask = ListedColormap(CMAP.rgba.colors[:2, :])
CMAP.rgb_mask = ListedColormap(CMAP.rgba.colors[:2, :3])

# %% Set up logger
logger = Logger(
		min_level_file=Logger.NAME2LEVEL.DEBUG if DEBUG else Logger.NAME2LEVEL.INFO,
		min_level_console=Logger.NAME2LEVEL.DEBUG if DEBUG else Logger.NAME2LEVEL.INFO,
)


# %% Check inputs
def check_inputs():
	if not os.path.exists(PATH_DATA):
		raise ValueError(f"Data path not found at {PATH_DATA}")
	if not os.path.exists(PATH_ILASTIK_EXE):
		raise ValueError(f"Ilastik executable not found at {PATH_ILASTIK_EXE}")

# check_inputs()
