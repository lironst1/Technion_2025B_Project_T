import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import IMAGE_EXTENSIONS, PATH_DATA, get_path, AUTO_CONTRAST, SIGMAS, RANDOM_FOREST_CLASSIFIER_KW, \
	PATH_ILASTIK_EXE, set_props_kw_image, CMAP, logger
from utils import flatten_image_tree, DataManager, print_image_tree, PixelClassifier
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
# dm.fit()
# file_pkl = dm.save_pixel_classifier()

file_pkl = os.path.join(get_path("all_head"), "pixel_classifier.pkl")
dm = DataManager(
		dir_root=get_path("2025_03_05"),
		# sample_size=10,
		# labeled=False,
		pixel_classifier=file_pkl
)
# dm.predict(plot=True, save_fig=True)
dm.plot_stats()

# run_ilastik(path_project=rel_path("MyProject.ilp"), dir_root=rel_path("train"))

# split_labels_tif(filename_labels=rel_path(r"train_2\labels.tif"),
# 		dir_labeled_images=rel_path(r"train_2\data"),
# 		dir_target=rel_path(r"all_head\labels")
# )

pass
