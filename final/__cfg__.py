import os
import numpy as np
from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_, ispc

# gr.update_rcParams("liron_utils-article")

# %% Constants
PATH_DATA = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data" if ispc \
	else "/Users/lironst/Library/CloudStorage/OneDrive-Technion/Homework/2025B/114252 - Project T/Data"  # Path to the data directory
rel_path = lambda *args: os.path.join(PATH_DATA, *args)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']

# %% Data Manager
HIST_EQUALIZE = True

# %% Pixel Classifier
SIGMAS = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0]
RANDOM_FOREST_CLASSIFIER_KW = dict_(
		n_estimators=100,
		max_depth=None,
		n_jobs=-2,
)

# %% Ilastik
PATH_ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe"  # Path to the Ilastik's 'run_ilastik.bat' script used for headless processing

# %% Napari

# %% Plots
set_props_kw_image = dict(axis="image", ticks=False, xy_lines=False)
EQUALIZE_ADAPTHIST_KW = dict(clip_limit=0.025)

CMAP = dict_(
		rgba=ListedColormap(np.array([
			# [R, G, B, alpha]
			[0.471, 0.145, 0.024, 0.1],  # background
			[0.357, 0.835, 0.973, 0.2],  # nuclei
			[0.573, 0.537, 0.910, 0.15],  # dirt
			[0.424, 0.008, 0.757, 0.15],  # inner dirt
		])),
)
CMAP.rgb = ListedColormap(CMAP.rgba.colors[:, :3])  # RGB colormap for plotting

# %% Set up logger
logger = Logger(min_level_file=Logger.NAME2LEVEL.INFO, min_level_console=Logger.NAME2LEVEL.INFO)


# %% Check inputs
def check_inputs():
	if not os.path.exists(PATH_DATA):
		raise ValueError(f"Data path not found at {PATH_DATA}")
	if not os.path.exists(PATH_ILASTIK_EXE):
		raise ValueError(f"Ilastik executable not found at {PATH_ILASTIK_EXE}")

# check_inputs()
