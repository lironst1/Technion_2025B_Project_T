import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import logger, set_props_kw_image
from utils import move_and_rename_files, select_random_images, run_ilastik, \
	run_ilastik_parallel, DataManager

# select_random_images(
# 		dir_root=r"C:\Users\liron\Downloads\Data\all",
# 		dir_target=r"C:\Users\liron\Downloads\Data",
# 		train=300,
# 		test=300
# )

# outputs = run_ilastik_parallel(
# 		ilastik_exe_path=r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe",
# 		project_path=r"C:\Users\liron\Downloads\Data\MyProject.ilp",
# 		data_path=r"C:\Users\liron\Downloads\Data\train",
# 		save_dir=r"C:\Users\liron\Downloads\Data\train\ilastik_output"
# )
#
# for idx, arr in enumerate(outputs):
# 	print(f"Image {idx}: shape {arr.shape}, dtype {arr.dtype}")

dm = DataManager(dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\test\input")

run_ilastik(path_project=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\MyProject.ilp",
		dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train"
)
pass
