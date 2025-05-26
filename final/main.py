import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import IMAGE_EXTENSIONS, PATH_DATA, get_path, HIST_EQUALIZE, SIGMAS, RANDOM_FOREST_CLASSIFIER_KW, \
	PATH_ILASTIK_EXE, set_props_kw_image, EQUALIZE_ADAPTHIST_KW, CMAP, logger
from utils import flatten_image_tree, DataManager, print_image_tree, PixelClassifier
from utils_napari import split_labels_tif

# flatten_image_tree(
# 		dir_root=rel_path("original"),
# 		dir_target=rel_path(r"all_head\data"),
# 		path_excel=rel_path("betacatenin_head.xlsx"),
# )
# print_image_tree(rel_path("all_head"))

dm = DataManager(dir_root=get_path("all_head"))
dm.fit()
file_pkl = dm.save_pixel_classifier()

dm = DataManager(
		dir_root=get_path("all_head"),
		sample_size=10,
		labeled=False,
		pixel_classifier=file_pkl
)
dm.predict(plot=True)

# run_ilastik(path_project=rel_path("MyProject.ilp"), dir_root=rel_path("train"))

# split_labels_tif(filename_labels=rel_path(r"train_2\labels.tif"),
# 		dir_labeled_images=rel_path(r"train_2\data"),
# 		dir_target=rel_path(r"all_head\labels")
# )

pass
