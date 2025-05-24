import os
from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_

# gr.update_rcParams("liron_utils-article")

# %% Constants
PATH_DATA = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data"  # Path to the data directory
rel_path = lambda *args: os.path.join(PATH_DATA, *args)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']

# Ilastik
PATH_ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe"  # Path to the Ilastik's 'run_ilastik.bat' script used for headless processing
ILASTIK_PROBS_EXTENSION = '_Probabilities_.npy'

# Napari


# Plots
set_props_kw_image = dict(axis="image", ticks=False, xy_lines=False)
EQUALIZE_ADAPTHIST_KW = dict(clip_limit=0.025)

CMAP = dict_(
		rgba=ListedColormap((
			# (R, G, B, alpha)
			(0.471, 0.145, 0.024, 0.25),  # background
			(0.357, 0.835, 0.973, 0.2),  # nuclei
			(0.573, 0.537, 0.910, 0.15),  # dirt
			(0.424, 0.008, 0.757, 0.2),  # inner dirt
		)),
)
CMAP.rgb = ListedColormap(tuple([rgba[:-1] for rgba in CMAP.rgba.colors]))

# %% Set up logger
logger = Logger(min_level_file=Logger.NAME2LEVEL.INFO, min_level_console=Logger.NAME2LEVEL.INFO)


# %% Check inputs
def check_inputs():
	if not os.path.exists(PATH_DATA):
		raise ValueError(f"Data path not found at {PATH_DATA}")
	if not os.path.exists(PATH_ILASTIK_EXE):
		raise ValueError(f"Ilastik executable not found at {PATH_ILASTIK_EXE}")


check_inputs()
