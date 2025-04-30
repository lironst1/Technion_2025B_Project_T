from matplotlib.colors import ListedColormap

from liron_utils import graphics as gr
from liron_utils.pure_python import Logger
from liron_utils.pure_python import dict_

# gr.update_rcParams("liron_utils-article")


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
