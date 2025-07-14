import os
import numpy as np
import tifffile
from natsort import natsorted
from tqdm import tqdm
# import napari

from __cfg__ import logger
from utils import is_image


# viewer.layers["Labels"].data.shape

# viewer = napari.Viewer()
# viewer.open(
#         path=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2\data",
#         layer_type="image",
#         name="train_2",
# )
# viewer.add_image()

# todo: add a way to open a random sample, label it, and save the labels back in the original path

def split_labels_tif(filename_labels, dir_labeled_images, dir_target):
	"""
	Split a 3D label TIF file into individual 2D slices and save them as separate TIF files.

	Parameters
	----------
	filename_labels :       str
		Path to the 3D label TIF file of shape (N, H, W), where N is the number of slices.
	dir_labeled_images :    str
		Directory containing the original images (not their labeling).
	dir_target :            str
		Directory where the individual label slices will be saved.

	Returns
	-------

	"""
	# Load the 3D label tif file
	labels_3d = tifffile.imread(filename_labels)

	# Get a sorted list of image filenames in the original image directory
	filenames = os.listdir(dir_labeled_images)
	filenames = natsorted([f.replace(".lnk", "") for f in filenames if is_image(f)])

	# Check that the number of slices matches the number of images
	if len(filenames) != labels_3d.shape[0]:
		raise ValueError(f"Number of slices in labels file ({labels_3d.shape[0]}) "
		                 f"does not match number of images ({len(filenames)}).")

	# Ensure output directory exists
	os.makedirs(dir_target, exist_ok=True)

	# Iterate over each slice
	for i, filename in tqdm(enumerate(filenames), total=len(filenames), desc="Processing slices"):
		label_slice = labels_3d[i]

		if np.any(label_slice):  # Save only if there are non-zero labels
			filename_out = os.path.join(dir_target, filename)
			if os.path.exists(filename_out):
				logger.warning(f"File {filename_out} already exists. Skipping.")
				continue
			tifffile.imwrite(filename_out, label_slice)

	logger.info("Label splitting completed.")
