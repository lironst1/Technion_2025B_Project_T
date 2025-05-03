import os
from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_

# gr.update_rcParams("liron_utils-article")


PATH_ILASTIK_EXE = r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe"  # Path to the Ilastik's 'run_ilastik.bat' script used for headless processing.

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']
PROBS_EXTENSION = '_Probabilities_.npy'

set_props_kw_image = dict(axis="image", ticks=False, xy_lines=False)
EQUALIZE_ADAPTHIST_KW = dict(clip_limit=0.025)

CMAP = dict_(
		rgb=ListedColormap([
			gr.hex2rgb(gr.COLORS.YELLOW),  # nuclei
			gr.hex2rgb(gr.COLORS.BLUE),  # background
			gr.hex2rgb(gr.COLORS.RED),  # dirt
		]),
		rgba=ListedColormap([
			gr.hex2rgb(gr.COLORS.YELLOW) + (0.25,),  # nuclei
			gr.hex2rgb(gr.COLORS.BLUE) + (0.2,),  # background
			gr.hex2rgb(gr.COLORS.RED) + (0.15,),  # dirt
		]),
)

logger = Logger(
		min_level_file=Logger.NAME2LEVEL.INFO,
		min_level_console=Logger.NAME2LEVEL.INFO
)


def check_inputs():
	if not os.path.exists(PATH_ILASTIK_EXE):
		raise ValueError(f"Ilastik executable not found at {PATH_ILASTIK_EXE}")


check_inputs()
