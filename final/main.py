import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import logger, set_props_kw_image, rel_path
from utils import flatten_image_tree, DataManager, print_image_tree
from utils_napari import split_labels_tif

flatten_image_tree(
		dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\original",
		dir_target=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\all",
		path_excel=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\betacatenin_head.xlsx",
)

dm = DataManager(
		dir_root=rel_path("all_head"),
		sample_size=0.1,
)

# todo: configure "all_head" to be the root directory, add label images and run pixel classification

# dm.copy_images(dir_target=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2")

# run_ilastik(path_project=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\MyProject.ilp",
# 		dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train"
# )

# split_labels_tif(filename_labels=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2\labels.tif",
# 		dir_labeled_images=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2\data",
# 		dir_target=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\all\labels")


# print_image_tree(r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\all")
print(os.getcwd())
pass
