from liron_utils import graphics as gr
from liron_utils.pure_python import Logger

gr.update_rcParams("liron_utils-article")

logger = Logger(
		min_level_file=Logger.NAME2LEVEL.INFO,
		min_level_console=Logger.NAME2LEVEL.INFO
)
