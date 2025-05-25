import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from liron_utils import graphics as gr

from __cfg__ import logger, set_props_kw_image, rel_path
from utils import flatten_image_tree, DataManager, print_image_tree, PixelClassifier
from utils_napari import split_labels_tif

# flatten_image_tree(
# 		dir_root=rel_path("original"),
# 		dir_target=rel_path(r"all_head\data"),
# 		path_excel=rel_path("betacatenin_head.xlsx"),
# )
# print_image_tree(rel_path("all_head"))


file_pkl = rel_path("pixel_classifier.pkl")
if os.path.exists(file_pkl):
	logger.info(f"Loading pixel classifier from {file_pkl}")
	with open(file_pkl, "rb") as f:
		pc = pkl.load(f)

else:  # train
	logger.info(f"Pixel classifier not found at {file_pkl}, creating a new one.")
	dm = DataManager(
			dir_root=rel_path("all_head"),
			sample_size=None,
			labeled=True
	)
	data = dm[:]
	images = [data[i].image for i in range(len(data))]
	labels = [data[i].labels for i in range(len(data))]
	pc = PixelClassifier()
	pc.fit(images=images, labels=labels)
	with open(file_pkl, "wb") as f:
		pkl.dump(pc, f)

# test
dm = DataManager(
		dir_root=rel_path("all_head"),
		sample_size=10,
		labeled=False
)
data = dm[:]
images = [data[i].image for i in range(len(data))]
labels = [data[i].labels for i in range(len(data))]
prob = pc.predict_prob(images=images)

for idx in range(len(dm)):
	axs = dm.plot(idx, prob=prob[idx])
	Ax = gr.Axes(axs=axs)
	Ax.set_props(**set_props_kw_image)

# run_ilastik(path_project=rel_path("MyProject.ilp"), dir_root=rel_path("train"))

# split_labels_tif(filename_labels=rel_path(r"train_2\labels.tif"),
# 		dir_labeled_images=rel_path(r"train_2\data"),
# 		dir_target=rel_path(r"all_head\labels")
# )

pass
