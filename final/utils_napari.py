import os
import numpy as np
import matplotlib.pyplot as plt
import napari
import tifffile
from natsort import natsorted
from tqdm import tqdm

from __cfg__ import IMAGE_EXTENSIONS, CMAP, logger
from utils import is_image

# viewer.layers["Labels"].data.shape

# viewer = napari.Viewer()
# viewer.open(
#         path=r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\train_2\data",
#         layer_type="image",
#         name="train_2",
# )
# viewer.add_image()


# todo: continue from here
# todo: perform labeling to several images, run pixel classification in napari and make sure it's good.
# todo: If so, perform statistics. If not, run from ilastik.


def split_labels_tif(filename_labels, dir_labeled_images, dir_target):
    # Load the 3D label tif file
    labels_3d = tifffile.imread(filename_labels)

    # Get sorted list of image filenames in the original image directory
    filenames_images = natsorted([f.replace(".lnk", "") for f in os.listdir(dir_labeled_images) if is_image(f)])

    # Check that the number of slices matches number of images
    if len(filenames_images) != labels_3d.shape[0]:
        raise ValueError(
                f"Number of slices in labels file ({labels_3d.shape[0]}) does not match number of images ({len(filenames_images)}).")

    # Ensure output directory exists
    os.makedirs(dir_target, exist_ok=True)

    # Iterate over each slice
    for i, filename in tqdm(enumerate(filenames_images), total=len(filenames_images), desc="Processing slices"):
        label_slice = labels_3d[i]

        if np.any(label_slice):  # Save only if there are non-zero labels
            output_path = os.path.join(dir_target, filename)
            if os.path.exists(output_path):
                logger.warning(f"File {output_path} already exists. Skipping.")
                continue
            tifffile.imwrite(output_path, label_slice)

    logger.info("Label splitting completed.")
