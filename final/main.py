import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import get_path
from utils import flatten_image_tree, print_image_tree
from utils_data_manager import DataManager
from utils_napari import split_labels_tif

# flatten_image_tree(
# 		dir_root=get_path("original"),
# 		dir_target=get_path("2025_03_05", "data"),
# 		date="2025_03_05",
# 		view=1,
# 		# path_excel=get_path("betacatenin_head.xlsx"),
# )
# print_image_tree(get_path("all_head"))

# dm = DataManager(dir_root=get_path("all_head"))
# dm.pixel_classifier_fit()
# file_pkl = dm.pixel_classifier_save()

# file_pkl = get_path("all_head", "pixel_classifier.pkl")
dm = DataManager(dir_root=get_path("2025_03_05", f"View1"),
		path_excel=get_path("betacatenin_head.xlsx"),
		date="2025_03_05", view=1)
# dm.plot_image_classification("2025_03_05__View1__7_beta_cat25X_TL_T77_C1")
dm.plot_stats(50, show_fig=True)
# dm.plot_frame(save_fig=True)

# run_ilastik(path_project=rel_path("MyProject.ilp"), dir_root=rel_path("train"))

# split_labels_tif(filename_labels=rel_path(r"train_2\labels.tif"),
# 		dir_labeled_images=rel_path(r"train_2\data"),
# 		dir_target=rel_path(r"all_head\labels")
# )


pass
