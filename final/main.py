import numpy as np
from sympy.stats import Probability

from liron_utils import graphics as gr

from __cfg__ import logger
from utils import *

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

images, probs = load_processed_data(dir_images=r"C:\Users\liron\Downloads\Data\test")
# todo: create data manager class (only load images and probs when needed)

Ax = gr.Axes(shape=(1, 3))
plot_predictions(Ax.axs.flatten(), im=images[0], prob=probs[0])
Ax.set_props(ax_title=["Image", "Probabilities", "Predictions"], grid=False, tick_labels=False)

pass
