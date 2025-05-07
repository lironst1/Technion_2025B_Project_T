import os
import random
import shutil
from glob import glob
from natsort import natsorted
from multiprocessing import Pool
import subprocess
import numpy as np
import tempfile

from tqdm import tqdm
import h5py
import tifffile
import skimage
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd

from liron_utils.files import copy
from liron_utils.pure_python import NUM_CPUS, parallel_map, parallel_threading, dict_, NamedQueue
from liron_utils.signal_processing import rescale
from liron_utils import graphics as gr

from __cfg__ import PATH_ILASTIK_EXE, set_props_kw_image, IMAGE_EXTENSIONS, PROBS_EXTENSION, CMAP, \
	logger, EQUALIZE_ADAPTHIST_KW
import tests


def is_image(filename):
	ext = os.path.splitext(filename)[1].lower()
	if ext == ".lnk":
		return is_image(filename.replace(".lnk", ""))
	if ext in IMAGE_EXTENSIONS:
		return True
	return False


def copy_and_rename_files(dir_root, dir_target=None, path_excel=None, overwrite=False, symlink=True):
	"""
	Move images from a directory tree to a single directory and rename them accordingly.
	Examples
	--------
	└── dir_root
    │   ├── ABC
    │   │   ├── abc
    │   │   │   ├── image1.png
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   ├── DEF
    │   │   ├── image1.png
    │   │   ├── image2.png

    └── dir_target
    │   ├── ABC__abc__image1.png
    │   ├── ABC__image1.png
    │   ├── ABC__image2.png
    │   ├── DEF__image1.png
    │   ├── DEF__image2.png


	Parameters
	----------
	dir_root :              str
		Path to the root directory containing the images.
	dir_target :            str
		Path to the target directory where the images will be moved. If not specified, dir_root is used.
	path_excel :            str
		Path to an Excel file containing additional data. If specified, the function will filter the images based on
		the data in the Excel file. The Excel file should contain columns: ["Date", "Pos", "final frame of beta catenin"]
		When provided, the directory tree is assumed to have the following structure (<> is a placeholder for the actual
		values and their format):
		└── dir_root
	    │   ├── <Date yyyy_mm_dd>
	    │   │   ├── View<Pos #>
	    │   │   │   ├── Max_C1 (doesn't have to appear)
	    │   │   │   │   ├── <image_name>.tif
	overwrite, symlink :    bool, optional
		See copy()

	Returns
	-------

	"""
	if dir_target is None:
		dir_target = dir_root

	excel_data = None
	if path_excel is not None:
		try:
			excel_data = pd.read_excel(path_excel)
		except PermissionError:
			logger.error(f"Permission error. Make sure to close the Excel file before running the script.")
			raise
		excel_data["Date"] = pd.to_datetime(excel_data["Date"], format="%d.%m.%y")
		excel_data["Date"] = excel_data["Date"].ffill()
		excel_data["Pos"] = excel_data["Pos"].astype(int, errors="ignore")
		excel_data["Pos"] = excel_data["Pos"].ffill()

	n_files = 0
	os.makedirs(dir_target, exist_ok=True)
	for dir_cur, _, filenames in os.walk(dir_root, topdown=False):
		filenames = natsorted([f for f in filenames if is_image(f)])
		if len(filenames) == 0:
			continue

		rel_path = os.path.relpath(dir_cur, dir_root)

		if excel_data is not None:  # Filter based on Excel data
			dir_cur_list = rel_path.split(os.sep)  # e.g., ['2025_01_29', 'View1', 'Max_C1']
			if len(dir_cur_list) < 2:
				continue  # should have ./<date>/<View#>

			dir_cur_data_date = pd.to_datetime(dir_cur_list[0], format="%Y_%m_%d")
			dir_cur_data_view = int(dir_cur_list[1].lower().split("view")[-1])

			# Check if the current date and view match any rows in the Excel data
			matching_rows = excel_data[excel_data["Date"] == dir_cur_data_date]
			if matching_rows.empty:
				logger.warning(f"No matching rows found in Excel file for "
				               f"date {dir_cur_data_date.strftime('%Y_%m_%d')}.")
				continue

			matching_rows = matching_rows[matching_rows["Pos"] == dir_cur_data_view]
			if matching_rows.empty:
				logger.warning(f"No matching rows found in Excel file for "
				               f"date {dir_cur_data_date.strftime('%Y_%m_%d')} and view {dir_cur_data_view}.")
				continue

			max_frame = int(matching_rows["final frame of beta catenin"].max())
			filenames = filenames[:max_frame + 1]

		copy(src=[os.path.join(dir_cur, f) for f in filenames],
				dst=[os.path.join(dir_target, f"{rel_path.replace(os.sep, '__')}__{f}")
					for f in filenames],  # e.g., 2025_01_29__View1__Max_C1__image1.tif,
				overwrite=overwrite,
				symlink=symlink)

		n_files += len(filenames)

	logger.info(f"Finished copying {n_files} file{' links' if symlink else 's'} into {dir_target}.")


def run_ilastik(path_project, dir_root, image_ext=".tif"):
	if not os.path.exists(path_project):
		raise ValueError(f"Project file not found at {path_project}")
	if not os.path.isdir(dir_root):
		raise ValueError(f"Path is not a directory: {dir_root}")
	if " " in path_project or " " in dir_root:
		# Ilastik doesn't like spaces in path_project and dir_root, so we'll need to create a symlink to them
		tests.test_symlink_admin_priv()
	if image_ext not in IMAGE_EXTENSIONS:
		raise ValueError(f"Invalid image extension: {image_ext}. Supported extensions are: {IMAGE_EXTENSIONS}")

	dir_out = os.path.join(dir_root, "output")
	os.makedirs(dir_out, exist_ok=True)

	with tempfile.TemporaryDirectory() as temp_dir:
		if " " in path_project:  # Ilastik doesn't like spaces in the path_project
			tmp = path_project
			path_project = os.path.join(temp_dir, os.path.basename(tmp))
			os.symlink(src=tmp, dst=path_project, target_is_directory=False)
			logger.info(f"Created symlink to {path_project}")

		if " " in dir_root:
			tmp = dir_root
			dir_root = os.path.join(temp_dir, os.path.basename(tmp))
			os.symlink(src=tmp, dst=dir_root, target_is_directory=True)
			logger.info(f"Created symlink to {dir_root}")

			tmp = dir_out
			dir_out = os.path.join(temp_dir, os.path.basename(tmp))
			os.symlink(src=tmp, dst=dir_out, target_is_directory=True)
			logger.info(f"Created symlink to {dir_out}")

		filenames_format = os.path.join(dir_root, f"*{image_ext}")
		output_filename_format = os.path.join(dir_out, "{nickname}.npy")

		command = [
			PATH_ILASTIK_EXE,
			f'--headless',
			f'--readonly',
			f'--input-axes=zyx',
			# f'--stack_along="c"',
			f'--export_source=probabilities',
			f'--project={path_project}',
			f'--output_format=numpy',
			f'--output_filename_format={output_filename_format}',
			f'{filenames_format}'
		]

		subprocess.run(command, check=True)

		"""
		TODO: Current error:
		ilastik.applets.base.applet.DatasetConstraintError: Constraint of 'Pixel Classification' applet was violated: All input images must have the same dimensionality.  Your new image has 4 dimensions (including channel), but your other images have 3 dimensions.
		ERROR 2025-05-04 01:26:14,752 log_exception 33984 23564 Project could not be loaded due to the exception shown above.
		Aborting Project Open Action
		"""
		pass


# 	with h5py.File(output_file, "r") as f:
# 		dataset_keys = list(f.keys())
# 		data = f[dataset_keys[0]][:]
#
# # Save output
# base_filename = os.path.splitext(os.path.basename(input_path))[0]
# if save_dir is not None:
# 	os.makedirs(save_dir, exist_ok=True)
# 	save_path = os.path.join(save_dir, f"{base_filename}.{save_format}")
# 	if save_format == "npy":
# 		np.save(save_path, data)
# 	elif save_format == "tif":
# 		tifffile.imwrite(save_path, data.astype(np.float32))  # or np.uint8 if needed
# 	else:
# 		raise ValueError(f"Unknown save_format: {save_format}")


def run_ilastik_parallel(
		path_project,
		filenames,
		dir_target=None,
		save_format="npy"
):
	"""
	Run ilastik headless on all images in a specified folder using parallel processing, providing
	capabilities for automatic saving and progress tracking. This function enables processing of
	multiple images in parallel with user-specified configurations such as the number of workers,
	file extensions, and output formats.

	Parameters
	----------
	path_project :      str
	    Path to the Ilastik project file (.ilp) for loading the pre-trained model.
	filenames :         list[str]
	    List of image filenames to be processed. If a directory is provided, all images in the directory will be processed.
	dir_target :        str, optional
	    Directory path for saving the output files. If not specified, results are not saved.
	save_format :       str, optional
	    Format for saving output files, either `"npy"` or `"tif"` (default is `"npy"`).

	Returns
	-------
	list[np.ndarray]
	    List of processed output arrays corresponding to the images in the specified folder.
	"""
	if not os.path.exists(path_project):
		raise ValueError(f"Project file not found at {path_project}")
	if save_format not in ["npy", "tif"]:
		raise ValueError(f"Invalid save format: {save_format}. Use 'npy' or 'tif'.")

	args_list = [(PATH_ILASTIK_EXE, path_project, f, dir_target, save_format) for f in filenames]

	# outputs = parallel_threading(
	# 		func=run_ilastik,
	# 		iterable=args_list,
	# 		)
	outputs = []
	# with Pool(processes=n_workers) as pool:
	# 	for result in tqdm(pool.imap(run_ilastik, args_list), total=len(args_list)):
	# 		outputs.append(result)
	for args in tqdm(args_list):
		outputs.append(run_ilastik(*args))

	logger.info("Finished processing all images.")
	return outputs


class DataManager:
	def __init__(self, dir_root, sample_size=None, n_max_in_memory=10):
		"""
		Data manager for loading images and associated data.
		
		Parameters
		----------
		dir_root :            str
			Path to the directory/tree containing images.
			Associated data is assumed to be in the same directory as each image, or in an ./output directory, e.g.:
			└── dir_root
		    │   ├── ABC
		    │   │   ├── image1.png
		    │   │   ├── ...
		    │   │   ├── output
		    │   │   │   ├── image1.npy
		    │   │   │   ├── ...
		sample_size :           int, float, or bool, optional
			Number of images to randomly sample from the directory.
			If given in the range (0, 1], it is interpreted as a fraction (True is the same as 1, i.e., use all data in
			random order. False will use all data in the order discovered by os.path.walk). If None, all images are used.
		n_max_in_memory :       int, optional
			Maximum number of images to keep in memory at once. If exceeded, the oldest image will be removed from
			memory. Default is 5.
		"""
		if not os.path.isdir(dir_root):
			raise ValueError(f"Image directory not found: {dir_root}")
		self.dir_root = dir_root

		# Discover image and data files
		filenames_images = []
		for dir_cur, _, filenames in os.walk(dir_root, topdown=False):
			for filename in filenames:
				if is_image(filename):
					rel_path = os.path.relpath(dir_cur, dir_root)
					if rel_path == ".":
						rel_path = ""
					filenames_images.append(os.path.join(rel_path, filename))

		# Random sampling
		if sample_size is not None and sample_size is not False:
			if sample_size is True:
				sample_size = len(filenames_images)
			elif 0 < sample_size <= 1:  # fraction
				sample_size = int(sample_size * len(filenames_images))
			elif sample_size > 1:  # integer
				if sample_size > len(filenames_images):
					raise ValueError(f"sample_size ({sample_size}) exceeds the "
					                 f"number of available images ({len(filenames_images)}).")
				if int(sample_size) != sample_size:
					raise ValueError(f"sample_size must be a non-negative float or bool (given {sample_size} instead).")
			else:
				raise ValueError(f"sample_size must be a non-negative float or bool (given {sample_size} instead).")

			filenames_images = random.sample(filenames_images, min(sample_size, len(filenames_images)))

		self.num_samples = len(filenames_images)

		# Load all associated data
		self.basenames = [os.path.splitext(f)[0] for f in filenames_images]
		self.paths = dict_()  # {<basename>: {<idx>, <path_image>, <path_asoc_data>}}
		for idx, filename in enumerate(filenames_images):
			basename = self.basenames[idx]

			path_asoc_data = None
			tmp = os.path.join(dir_root, basename + PROBS_EXTENSION)
			if os.path.exists(tmp):
				path_asoc_data = tmp
			tmp = os.path.join(dir_root, os.path.dirname(basename), "output",
					os.path.split(basename)[1] + PROBS_EXTENSION)
			if os.path.exists(tmp):
				path_asoc_data = tmp

			self.paths[basename] = dict_(
					idx=idx,
					path_image=os.path.join(dir_root, filename),
					path_asoc_data=path_asoc_data,
			)

		# Cache filename -> (image, data)
		self.cache = NamedQueue(max_size=n_max_in_memory)

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		"""Get the image and associated data for a given index."""
		if isinstance(idx, int):
			basename = self.basenames[idx]
		elif isinstance(idx, str):
			basename = idx
			idx = self.paths[basename].idx
		elif isinstance(idx, slice):
			return [self[i] for i in range(self.num_samples)[idx]]
		elif isinstance(idx, list):
			return [self[i] for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		if idx < 0 or idx >= self.num_samples:
			raise IndexError(f"Index {idx} out of range (0-{self.num_samples - 1})")

		if basename in self.cache:  # Check if the image is already in memory
			data = self.cache[basename]
			image = data.image
			asoc_data = data.asoc_data
			logger.debug(f"Image {idx}:{basename} loaded from cache. Cache size: {len(self.cache)}.")

		else:  # Load the image and associated data
			image = self.imread(self.paths[basename].path_image)
			if self.paths[basename].path_asoc_data is None:
				asoc_data = None
			else:
				asoc_data = np.load(self.paths[basename].path_asoc_data)
			data = dict_(image=image, asoc_data=asoc_data)
			self.cache.enqueue(name=basename, item=data)
			logger.debug(f"Image {idx}:{basename} loaded to cache. Cache size: {len(self.cache)}.")

		return image, asoc_data

	def has_asoc_data(self, idx):
		"""Check if the image has associated data."""
		if isinstance(idx, int):
			basename = self.basenames[idx]
		elif isinstance(idx, str):
			basename = idx
		elif isinstance(idx, slice):
			return [self.has_asoc_data(i) for i in range(self.num_samples)[idx]]
		elif type(idx) in [list, tuple, np.ndarray]:
			return [self.has_asoc_data(i) for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		return self.paths[basename].path_asoc_data is not None

	def copy_images(self, dir_target):
		"""
		Copy images to a target directory.

		Parameters
		----------
		dir_target : str
			Path to the target directory where the images will be copied.

		Returns
		-------

		"""
		src = [os.path.join(self.dir_root, self.paths[basename].path_image) for basename in self.paths]
		copy(src=src, dst=dir_target)

	@staticmethod
	def imread(filename):
		image = plt.imread(filename)
		image = skimage.exposure.equalize_adapthist(image, **EQUALIZE_ADAPTHIST_KW)
		return image

	def plot(self, idx, axs=None, which="all", **kwargs):
		"""
		Plot predictions.

		Parameters
		----------
		idx :           int
			Index of the image to plot.
		axs :           Axes or list[Axes]
			Axes object to plot on. If given as a list, plots will be
			[image, probabilities, predictions, image+probabilities] (or a part, depending on len(axs)).
			If None, a new Axes object will be created.
		which :         str or list[str]
			Which plots to show:
			- "image": Show the image.
			- "probabilities": Show the predicted probabilities.
			- "predictions": Show the predicted class labels.
			- "image+probabilities": Show the image and predicted probabilities together.
			- "all": Show all plots [image, probabilities, predictions, image+probabilities] (or a part).
		**kwargs :      sent to imshow

		Returns
		-------

		"""
		if not isinstance(idx, int) and not isinstance(idx, str):
			raise ValueError("Using iterable indices is not supported. Please call plot() for each index separately.")
		if "cmap" in kwargs:
			raise ValueError("cmap is not supported. Define 'cmap' in the __cfg__ file instead.")

		image, asoc_data = self[idx]
		prob = asoc_data

		WHICH_VALUES = ["image", "probabilities", "predictions", "image+probabilities"]
		if isinstance(which, str):
			if which == "all":
				if prob is None:
					which = ["image"]
				else:
					which = WHICH_VALUES
			else:
				which = [which]
		if not set(which).issubset(set(WHICH_VALUES)):
			raise ValueError(f"Invalid value in 'which'. Allowed values are: {WHICH_VALUES} or 'all'.")

		if axs is None:  # create new axes
			shape = (1, len(which))
			if len(which) == 4:  # len(WHICH_VALUES)
				shape = (2, 2)
			axs = gr.Axes(shape=shape)
		if isinstance(axs, Axes):
			axs = [axs]
		elif isinstance(axs, gr.Axes):
			axs = axs.axs.flatten()

		def plot_image(ax, image, cmap="gray", **kwargs):
			ax.imshow(image, cmap=cmap, **kwargs)

		def plot_probabilities(ax, prob, cmap, **kwargs):
			X = np.einsum("...i,ij->...j", prob, np.array(CMAP.rgb.colors))
			ax.imshow(X, cmap=cmap, **kwargs)

		def plot_predictions(ax, prob, cmap, **kwargs):
			X = prob.argmax(axis=-1)
			ax.imshow(X, cmap=cmap, **kwargs)

		for i, ax in enumerate(axs):
			if i >= len(which):
				break

			if which[i] == "image":
				plot_image(ax, image, **kwargs)
			else:
				if prob is None:
					logger.warning(f"Image {idx} has no associated data. Skipping {which[i]} plot.")
					continue

				if which[i] == "probabilities":
					plot_probabilities(ax, prob, cmap=CMAP.rgb, **kwargs)
				elif which[i] == "predictions":
					plot_predictions(ax, prob, cmap=CMAP.rgb, **kwargs)
				else:  # which[i] == "image+probabilities":
					plot_image(ax, image, **kwargs)
					plot_probabilities(ax, prob, cmap=CMAP.rgba, **kwargs)

		return axs
