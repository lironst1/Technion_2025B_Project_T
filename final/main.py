import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import logger, set_props_kw_image
from utils import move_and_rename_files, select_random_images, run_ilastik_single_image, \
	run_ilastik_on_folder_parallel, imread, load_processed_data, DataManager

# select_random_images(
# 		dir_root=r"C:\Users\liron\Downloads\Data\all",
# 		dir_target=r"C:\Users\liron\Downloads\Data",
# 		train=300,
# 		test=300
# )

# outputs = run_ilastik_on_folder_parallel(
# 		ilastik_exe_path=r"C:\Program Files\ilastik-1.4.1rc2-gpu\ilastik.exe",
# 		project_path=r"C:\Users\liron\Downloads\Data\MyProject.ilp",
# 		data_path=r"C:\Users\liron\Downloads\Data\train",
# 		save_dir=r"C:\Users\liron\Downloads\Data\train\ilastik_output"
# )
#
# for idx, arr in enumerate(outputs):
# 	print(f"Image {idx}: shape {arr.shape}, dtype {arr.dtype}")

# images, probs, filenames = load_processed_data(path_images=r"C:\Users\liron\Downloads\Data\test")
# # todo: create data manager class (only load images and probs when needed)
#
# for i in tqdm(range(len(filenames))):
# 	filename = filenames[i]
# 	save_file_name = os.path.join(os.path.split(filename)[0], "figs",
# 			os.path.splitext(os.path.basename(filename))[0] + ".png")
# 	if os.path.exists(save_file_name):
# 		logger.info(f"i={i}: File {save_file_name} already exists, skipping.")
# 		continue
#
# 	image = images[i]
# 	prob = probs[i]
#
# 	Ax = gr.Axes(shape=(2, 2))
# 	plot_predictions([Ax.axs[0, 0], Ax.axs[1, 0], Ax.axs[1, 1]], im=image, prob=prob)
# 	plot_predictions(Ax.axs[0, 1], im=image, prob=prob)
# 	Ax.set_props(
# 			sup_title=os.path.splitext(os.path.basename(filename))[0],
# 			ax_title=["Image", "Probabilities", "Image + Probabilities", "Predictions"],
# 			save_file_name=save_file_name,
# 			show_fig=False,
# 			**set_props_kw_image)
# 	plt.close(Ax.fig)

dm = DataManager(dir_root=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\test\input")
dm.plot(0)
pass
