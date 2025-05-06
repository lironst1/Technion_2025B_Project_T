import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import logger, set_props_kw_image
from utils import copy_and_rename_files, select_random_images, run_ilastik, \
	run_ilastik_parallel, DataManager

# select_random_images(
# 		dir_root=r"C:\Users\liron\Downloads\Data\all",
# 		dir_target=r"C:\Users\liron\Downloads\Data",
# 		train=300,
# 		test=300
# )

# copy_and_rename_files(
# 		dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\original",
# 		dir_target=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\all",
# 		# path_excel=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\betacatenin_head.xlsx",
# )

dm = DataManager(
		dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\all_head",
		sample_size=0.1,
)

dm.copy_images(dir_target=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2")

# run_ilastik(path_project=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\MyProject.ilp",
# 		dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train"
# )

pass
