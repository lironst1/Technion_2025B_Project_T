import os
import time
import warnings
import random
import functools
from collections.abc import Iterable
from collections import defaultdict
from natsort import natsorted
from prettytable import PrettyTable
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tifffile
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
from scipy.spatial import distance_matrix

from liron_utils.files import copy
from liron_utils.pure_python import dict_, NamedQueue, parallel_threading, tqdm_
from liron_utils.files import open_file, mkdirs
from liron_utils import graphics as gr

from __cfg__ import logger, IMAGE_EXTENSIONS, AUTO_CONTRAST, SIGMAS, RANDOM_FOREST_CLASSIFIER_KW, \
	set_props_kw_image, LABELS, LABELS2IDX, CMAP, DATA_TYPES, CPSAMEvalOut, Stats, CPSAM_EVAL_KW, TQDM_KW
import tests


def is_image(filename):
	ext = os.path.splitext(filename)[1].lower()
	if ext == ".lnk":
		return is_image(filename.replace(".lnk", ""))
	if ext in IMAGE_EXTENSIONS:
		return True
	return False


def print_image_tree(dir_root):
	"""
    Print the directory tree under `dir_root`, showing the number of images and labels in each logical group.

    Parameters
    ----------
    dir_root :      str or Path
        Path to the parent directory containing 'data', 'labels' and 'prob' subdirectories.
    """
	dir_data = os.path.join(dir_root, "data")
	dir_labels = os.path.join(dir_root, "labels")

	tests.dir_exist(dir_data)
	tests.dir_exist(dir_labels)

	# Collect image and label file paths
	files_data = natsorted([f.replace(".lnk", "") for f in os.listdir(dir_data) if is_image(f)])
	files_labels = natsorted([f.replace(".lnk", "") for f in os.listdir(dir_labels) if is_image(f)])

	# Build maps from logical group (e.g., ABC, ABC__abc) to counts
	image_counts = defaultdict(int)
	label_counts = defaultdict(int)

	def get_logical_group(filename):
		parts = filename.split("__")
		if len(parts) > 1:
			return "__".join(parts[:-1])
		return "."

	for f in files_data:
		group = get_logical_group(f)
		image_counts[group] += 1

	for f in files_labels:
		group = get_logical_group(f)
		label_counts[group] += 1

	# Print header
	field_names = ["Experiment", "Images", "Labels"]
	table = PrettyTable(field_names=field_names)
	table.align["Experiment"] = "l"
	table.align["Images"] = "r"
	table.align["Labels"] = "r"

	total_images = 0
	total_labels = 0

	all_keys = natsorted(set(image_counts) | set(label_counts))
	for key in all_keys:
		count_image = image_counts[key]
		count_label = label_counts[key]
		table.add_row([key, count_image if count_image > 0 else "", count_label if count_label > 0 else ""])
		total_images += count_image
		total_labels += count_label

	# Add total row
	table.add_row(["-" * len(field) for field in field_names])
	table.add_row(["TOTAL", total_images, f"{total_labels} ({total_labels / total_images:.2%})"])

	print(table)


def flatten_image_tree(dir_root, dir_target=None, path_excel=None, date=None, view=None, overwrite=False, symlink=True):
	"""
	Move images from a directory tree to a single directory and rename them accordingly.

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
    │   ├── ABC__abc__image1.png.lnk
    │   ├── ABC__image1.png.lnk
    │   ├── ABC__image2.png.lnk
    │   ├── DEF__image1.png.lnk
    │   ├── DEF__image2.png.lnk

	Examples
	--------
	>>> flatten_image_tree(dir_root='./original', dir_target='./all_head', path_excel='./betacatenin_head.xlsx')

	Parameters
	----------
	dir_root :              str
		Path to the root directory containing the images.
	dir_target :            str, optional
		Path to the target directory where the images will be moved. If not specified, dir_root is used.
	path_excel :            str, optional
		Path to an Excel file. If specified, the function will filter the images based on the data in the Excel file.
		The Excel file should contain columns: ["Date", "Pos", "final frame of beta catenin"]. When provided, the directory
		tree is assumed to have the following structure (<> is a placeholder for the actual values and their format):
		└── dir_root
	    │   ├── <Date yyyy_mm_dd>
	    │   │   ├── View<Pos #>
	    │   │   │   ├── Max_C1 (doesn't have to appear)
	    │   │   │   │   ├── <image_name>.tif
	date :                str, optional
		Date to filter images by. If provided, only images from this date will be copied.
	view :                int, optional
		View number to filter images by. If provided, only images from this view will be copied.
	overwrite, symlink :    bool, optional
		See copy()

	Returns
	-------

	"""
	if dir_target is None:
		dir_target = dir_root

	excel_data = None
	if path_excel is not None:
		tests.excel_permissions(path_excel)  # Check if the user has permissions to access the Excel file
		excel_data = pd.read_excel(path_excel)
		excel_data["Date"] = pd.to_datetime(excel_data["Date"], format="%d.%m.%y")
		excel_data["Date"] = excel_data["Date"].ffill()
		excel_data["Pos"] = excel_data["Pos"].astype(int, errors="ignore")
		excel_data["Pos"] = excel_data["Pos"].ffill()

	n_files = 0
	os.makedirs(dir_target, exist_ok=True)
	for dir_cur, _, filenames in tqdm(os.walk(dir_root, topdown=False), desc="Copying files", unit="files"):
		filenames = natsorted([f for f in filenames if is_image(f)])
		if len(filenames) == 0:
			continue

		rel_path = os.path.relpath(dir_cur, dir_root)

		dir_cur_list = rel_path.split(os.sep)  # e.g., ['2025_01_29', 'View1', 'Max_C1']
		if len(dir_cur_list) < 2:
			continue  # should have ./<date>/<View#>

		dir_cur_data_date = pd.to_datetime(dir_cur_list[0], format="%Y_%m_%d")
		dir_cur_data_view = int(dir_cur_list[1].lower().split("view")[-1])

		if date is not None:  # Filter based on date
			if isinstance(date, str):
				date = pd.to_datetime(date, format="%Y_%m_%d")
			if dir_cur_data_date != date:
				continue

		if view is not None:  # Filter based on view
			if isinstance(view, str):
				view = int(view.lower().split("view")[-1])
			if dir_cur_data_view != view:
				continue

		if excel_data is not None:  # Filter based on Excel data
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
	open_file(dir_target)  # Open the target directory in File Explorer


class RandomForestPixelClassifier:
	def __init__(self, sigmas=SIGMAS, **random_forest_classifier_kw):
		"""
		A pixel classification model that mimics ilastik's Random Forest-based approach.

		Features are extracted at multiple Gaussian scales, including:
		- Gaussian smoothed intensity
		- Gradient magnitude
		- Local variance (as a proxy for texture)

		The classifier is trained using user-provided annotations, and outputs per-pixel
		class probabilities and segmentation maps.

		Parameters
		----------
		sigmas : list[float], optional
		    Standard deviations for Gaussian smoothing used in feature extraction.
		"""
		random_forest_classifier_kw = RANDOM_FOREST_CLASSIFIER_KW | random_forest_classifier_kw

		self.sigmas = sigmas
		self.clf = RandomForestClassifier(**random_forest_classifier_kw)
		self.scaler = StandardScaler()
		self.feature_names = []

	def _compute_features(self, image):
		"""
        Extracts multiscale features from a 2D image.

        Parameters
        ----------
        image : ndarray
            Grayscale 2D image.

        Returns
        -------
        feature_stack : ndarray
            Array of shape (H, W, F) where F is the number of features per pixel.
        """
		features = []
		self.feature_names = []

		for sigma in self.sigmas:
			# Gaussian smoothed intensity
			gauss = gaussian_filter(image, sigma=sigma)
			features.append(gauss)
			self.feature_names.append(f'gaussian_{sigma}')

			# Gradient magnitude
			grad_mag = gaussian_gradient_magnitude(input=image, sigma=sigma)
			features.append(grad_mag)
			self.feature_names.append(f'gradient_magnitude_{sigma}')

		# Texture (Local binary pattern)
		# Note: LBP is typically applied to 2D grayscale images
		# For simplicity, we'll use a basic texture measure: variance in a window
		# size = int(2 * np.ceil(3 * sigma) + 1)
		# local_var = windowed_histogram(image=image.astype(np.uint8), footprint=np.ones((size, size)))
		# features.append(local_var)
		# self.feature_names.append(f'local_variance_{sigma}')

		# Stack features into a (H, W, F) array
		feature_stack = np.stack(features, axis=-1)
		return feature_stack

	def fit(self, images, labels):
		"""
		Train the classifier using multiple images and corresponding label masks.

		Parameters
		----------
		images : list[np.ndarray]
		    List of 2D grayscale images.
		labels : list[np.ndarray]
		    List of 2D label masks. Unlabeled pixels should have value 0.
		"""
		if not isinstance(images, list):
			images = [images]
			labels = [labels]

		X_all, y_all = [], []
		for image, label in tqdm(zip(images, labels), desc="Training PixelClassifier", unit="image", total=len(images)):
			features = self._compute_features(image)
			is_labeled = (label > 0)
			X = features[is_labeled]
			y = label[is_labeled]
			X_all.append(X)
			y_all.append(y)

		X_concat = np.concatenate(X_all, axis=0)
		y_concat = np.concatenate(y_all, axis=0)

		X_scaled = self.scaler.fit_transform(X_concat)
		self.clf.fit(X_scaled, y_concat)

	def predict_prob(self, images):
		"""
		Predict class probabilities for each pixel in each image.

		Parameters
		----------
		images : list[np.ndarray]
		    2D grayscale image or list of such images.

		Returns
		-------
		List of 3D arrays, each of shape (H, W, C).
		"""
		if not isinstance(images, list):
			images = [images]

		prob = []
		for image in images:
			features = self._compute_features(image)
			H, W, N = features.shape
			X_flat = features.reshape(-1, N)
			X_scaled = self.scaler.transform(X_flat)
			p = self.clf.predict_proba(X_scaled)
			p = p.reshape(H, W, self.clf.n_classes_)
			prob.append(p)

		return prob if len(prob) > 1 else prob[0]

	def predict(self, images):
		"""
		Predict the most likely class for each pixel in each image.

		Parameters
		----------
		images : list[np.ndarray]
		    2D grayscale image or list of such images.

		Returns
		-------
		List of 2D predicted class label arrays.
		"""
		prob = self.predict_prob(images)
		if isinstance(prob, list):
			return [np.argmax(p, axis=-1) + 1 for p in prob]
		else:
			return np.argmax(prob, axis=-1) + 1


class DataManager:
	def __init__(self, dir_root, sample_size=None, cache_size=20, labeled=None, random_forest_pixel_classifier=None):
		"""
		Data manager for loading images and associated data.
		
		Parameters
		----------
		dir_root :            str
			Path to the directory/tree containing images.
			Associated data is assumed to be in the same directory as each image, or in an ./prob directory, e.g.:
		    └── all
			│   ├── data (links to images, flattened to a single directory)
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif.lnk
			│   │   ├── ...
			│   ├── labels
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif
			│   │   ├── ...
			│   ├── random_forest_prob
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.pkl
			│   │   ├── ...
			│   ├── cpsam_out
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.pkl
			│   │   ├── ...
		sample_size :           int, float, or bool, optional
			Number of images to randomly sample from the directory.
			If given in the range (0, 1], it is interpreted as a fraction (True is the same as 1, i.e., use all data in
			random order. False will use all data in the order discovered by os.path.walk). If None, all images are used.
		cache_size :            int, optional
			Maximum number of images to keep in memory at once. If exceeded, the oldest image will be removed from
			memory. Default is 5.
		labeled :               bool or None, optional
			If True, only labeled images will be loaded.
			If False, only unlabeled images will be loaded.
			If None, both labeled and unlabeled images will be loaded.
		random_forest_pixel_classifier :      PixelClassifier or str, optional
			Path to pixel classifier .pkl file or the object itself.
			If None, a new pixel classifier will be created.
		"""
		tests.dir_exist(dir_root)
		self.dir_root = dir_root

		# Discover image and data files
		filenames_images = []
		for dir_cur, _, filenames in os.walk(os.path.join(dir_root, "data"), topdown=False):
			for filename in filenames:
				if is_image(filename):
					rel_path = os.path.relpath(dir_cur, os.path.join(dir_root, "data"))
					if rel_path == ".":
						rel_path = ""
					filename_image = os.path.join(rel_path, filename)
					if labeled is not None:  # Filter images based on labeled status
						filename_label = os.path.join(dir_root, "labels", filename)
						if labeled is True and not os.path.exists(filename_label) or \
								labeled is False and os.path.exists(filename_label):
							continue
					filenames_images.append(filename_image)

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
		else:
			filenames_images = natsorted(filenames_images)

		self.num_samples = len(filenames_images)

		# Get paths to images and associated data
		self.basenames: list[str] = [os.path.splitext(f)[0] for f in filenames_images]

		self.paths = dict_()  # {<basename>: {<idx>, <path_image>, <path_labels>, ...}}
		for idx, filename in enumerate(filenames_images):
			basename = self.basenames[idx]

			self.paths[basename] = dict_(idx=idx, **dict(zip(DATA_TYPES.keys(), len(DATA_TYPES) * [None])))
			for data_type, d in DATA_TYPES.items():
				path_data_type = os.path.join(dir_root, d.dirname, basename + d.ext)
				if os.path.exists(path_data_type):
					self.paths[basename][data_type] = path_data_type

		# Cache basename -> data
		self.cache = NamedQueue(max_size=cache_size)

		# Pixel classifier
		self.pixel_classifier = None
		if isinstance(random_forest_pixel_classifier, RandomForestPixelClassifier):
			self.pixel_classifier = random_forest_pixel_classifier
		elif isinstance(random_forest_pixel_classifier, str):
			tests.file_exist(random_forest_pixel_classifier)
			self.pixel_classifier = self._pickle_load(random_forest_pixel_classifier)
			logger.info(f"Loaded pixel classifier from {random_forest_pixel_classifier}.")

		# Cellpose Model (CPSAM)
		self.cpsam = None

		self._is_tqdm_running = False

	def __len__(self):
		return self.num_samples

	def __repr__(self):
		def num_labels():
			return np.count_nonzero(self.has_labels())

		def num_prob():
			return np.count_nonzero(self.has_prob())

		return (f"DataManager(#samples={self.num_samples}, #labels={num_labels()}, #prob={num_prob()}, "
		        f"dir_root={self.dir_root})")

	@staticmethod
	def _iterable_idx(tqdm_kw=None, use_threading=False, batch_size=1, shuffle=False):
		if tqdm_kw is None:
			tqdm_kw = dict(disable=True)
		tqdm_kw = TQDM_KW | tqdm_kw  # default tqdm settings

		if use_threading and batch_size > 1:
			raise ValueError("Threading cannot be used with `batch_size`>1.")

		def decorator(func):

			@functools.wraps(func)
			def wrapper(self, idx=None, *args, **kwargs):

				if idx is None:
					idx = list(range(self.num_samples))

				if isinstance(idx, int):
					return func(self, idx, *args, **kwargs)

				elif isinstance(idx, str):
					basename = idx
					idx = self.paths[basename].idx
					return func(self, idx, *args, **kwargs)

				elif isinstance(idx, slice) or isinstance(idx, Iterable):
					if isinstance(idx, slice):
						idx = list(range(self.num_samples)[idx])
					else:  # if isinstance(idx, Iterable)
						for i in range(len(idx)):
							if isinstance(idx[i], str):
								basename = idx[i]
								idx[i] = self.paths[basename].idx
							elif not isinstance(idx[i], int):
								raise TypeError(f"Index must be an integer or a string (given {type(idx[i])} instead).")

					if shuffle:
						random.shuffle(idx)

					idx_batch = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]

					tqdm_kw_ = dict(total=len(idx_batch),
							postfix=lambda i: dict(idx=i, basename=self.basenames[i])) | tqdm_kw
					if not tqdm_kw["disable"]:
						self._is_tqdm_running = True
						logger.info(f'{tqdm_kw_["desc"]} (total={tqdm_kw_["total"]})...', stacklevel=4)

					if use_threading:
						def func_threading(i):
							return func(self, i, *args, **kwargs)

						out = parallel_threading(func=func_threading, iterable=idx, tqdm_kw=tqdm_kw_)
					else:
						out = []
						for i in tqdm_(idx if batch_size == 1 else idx_batch, **tqdm_kw_):
							out.append(func(self, i, *args, **kwargs))

					if not tqdm_kw["disable"]:
						self._is_tqdm_running = False
						logger.info(f'Finished {tqdm_kw_["desc"].lower()} (total={len(out)}).', stacklevel=4)

					return out

				else:
					raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

			return wrapper

		return decorator

	@staticmethod
	def _imread(filename, auto_contrast=False):
		"""Read an image from a file."""

		def fix_contrst(image, low_clip_percent=1, high_clip_percent=0.015):
			hist = cv2.calcHist([image], [0], None, [2 ** 16], [0, 2 ** 16])
			accumulator = hist.cumsum()

			total = accumulator[-1]
			low_clip_value = total * low_clip_percent / 100.0
			high_clip_value = total * (1 - high_clip_percent / 100.0)

			minimum_gray = np.searchsorted(accumulator, low_clip_value)
			maximum_gray = np.searchsorted(accumulator, high_clip_value)

			# Avoid divide-by-zero
			if maximum_gray == minimum_gray:
				return cv2.convertScaleAbs(image)

			alpha = 255 / (maximum_gray - minimum_gray)
			beta = -minimum_gray * alpha

			out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
			return out

		image = tifffile.imread(filename)
		if auto_contrast:
			# image = equalize_adapthist(image, **EQUALIZE_ADAPTHIST_KW)
			image = fix_contrst(image)

		return image

	@staticmethod
	def _imwrite(image, filename):
		"""Write an image to a file."""
		mkdirs(os.path.dirname(filename))

		ext = os.path.splitext(filename)[1].lower()
		if ext == ".tif":
			tifffile.imwrite(filename, image)
		else:
			raise ValueError(f"Unsupported file extension for writing: {ext}.")

	@staticmethod
	def _pickle_load(filename):
		"""Load a pickle file."""
		with open(filename, "rb") as f:
			return pickle.load(f)

	@staticmethod
	def _pickle_dump(obj, filename):
		"""Dump an object to a pickle file."""
		mkdirs(os.path.dirname(filename))
		with open(filename, "wb") as f:
			pickle.dump(obj, f)

	def __getitem__(self, idx):
		"""Get the image and associated data for a given index."""

		if isinstance(idx, int):
			basename = self.basenames[idx]
		elif isinstance(idx, str):
			basename = idx
			idx = self.paths[basename].idx
		elif isinstance(idx, slice):
			return [self[i] for i in range(self.num_samples)[idx]]
		elif isinstance(idx, Iterable):
			return [self[i] for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		if idx < 0 or idx >= self.num_samples:
			raise IndexError(f"Index {idx} out of range (0-{self.num_samples - 1})")

		if basename in self.cache:  # Check if the image is already in memory
			data = self.cache[basename]

			logger.debug(f"Image {idx}:{basename} loaded from cache. Cache size: {len(self.cache)}.")

		else:  # Load the image and associated data
			data = dict_(basename=basename)

			for data_type in DATA_TYPES:
				data[data_type] = self.get_data(idx=idx, data_type=data_type)

			# Add to cache
			self.cache.enqueue(name=basename, item=data)

			logger.debug(f"Image {idx}:{basename} loaded to cache. Cache size: {len(self.cache)}.")

		return data

	@_iterable_idx()
	def get_data(self, idx=None, data_type=None):
		"""
		Get the image or associated data for a given index.

		Parameters
		----------
		idx : int, str, slice, Iterable[int], Iterable[str]
			Index of the image to get.
		data_type : str
			Type of data to return.

		Returns
		-------

		"""
		basename = self.basenames[idx]

		if data_type not in DATA_TYPES:
			raise ValueError(f"Invalid data type '{data_type}'. Valid types are: {', '.join(DATA_TYPES.keys())}.")

		if data_type == "basename":
			return basename

		elif data_type == "image":
			image = self._imread(self.paths[basename].image, auto_contrast=AUTO_CONTRAST)
			return image

		elif data_type == "labels":
			if self.paths[basename].labels is None:
				return None
			labels = self._imread(self.paths[basename].labels).round().astype("uint8")
			return labels

		elif data_type == "prob":
			if self.paths[basename].prob is None:
				return None
			prob = self._pickle_load(self.paths[basename].prob)
			return prob

		elif data_type == "cpsam_out":
			if self.paths[basename].cpsam_out is None:
				return None
			cpsam_out = self._pickle_load(self.paths[basename].cpsam_out)
			return cpsam_out

		else:
			raise ValueError(f"Invalid data type '{data_type}'. Valid types are: {', '.join(DATA_TYPES.keys())}.")

	@_iterable_idx()
	def has_labels(self, idx=None):
		"""Check if the image has labels."""
		basename = self.basenames[idx]

		return self.paths[basename].labels is not None

	@_iterable_idx()
	def has_prob(self, idx=None):
		"""Check if the image has associated probabilities."""
		basename = self.basenames[idx]

		return self.paths[basename].prob is not None

	def get_labeled_data(self):
		"""Get all images with associated labels."""
		return [self[idx] for idx in range(self.num_samples) if self.has_labels(idx)]

	def copy_images(self, dir_target, symlink=True):
		"""
		Copy images to a target directory.

		Parameters
		----------
		dir_target : str
			Path to the target directory where the images will be copied.
		symlink : bool, optional
			If True, create symbolic links to the images instead of copying them.

		Returns
		-------

		"""

		logger.info(f"Copying images to {dir_target}...")

		for data_type, d in DATA_TYPES.items():
			src = [os.path.join(self.dir_root, self.paths[basename][data_type]) for basename in self.paths if
				self.paths[basename][data_type] is not None]
			copy(src=src, dst=os.path.join(dir_target, d.dirname), symlink=symlink)

		logger.info(f"Images copied to {dir_target}.")

	@_iterable_idx(tqdm_kw=dict(desc="Saving images", unit="image"))
	def _save_images(self, idx=None, dir_target=None):
		"""
		Save images to a target directory.

		Parameters
		----------
		idx : int, str, slice, Iterable[int], Iterable[str]
			Index of the image to save.
		dir_target : str
			Path to the target directory where the images will be saved.

		Returns
		-------

		"""
		if dir_target is None:
			dir_target = os.path.join(self.dir_root, "figs", "_images")

		mkdirs(dir_target)

		basename = self.basenames[idx]
		image = self.get_data(idx=idx, data_type="image")
		self._imwrite(image=image, filename=os.path.join(dir_target, basename + DATA_TYPES.image.ext))

	def pixel_classifier_fit(self):
		idx = [i for i in range(self.num_samples) if self.has_labels(i)]

		images = self.get_data(idx=idx, data_type="image")
		labels = self.get_data(idx=idx, data_type="labels")

		if self.pixel_classifier is None:
			self.pixel_classifier = RandomForestPixelClassifier()
			logger.info(f"Creating new pixel classifier.")

		self.pixel_classifier.fit(images=images, labels=labels)
		return None

	def pixel_classifier_save(self, filename=None):
		"""Save the pixel classifier to a file."""
		if filename is None:
			filename = os.path.join(self.dir_root, "pixel_classifier.pkl")
		self._pickle_dump(self.pixel_classifier, filename)
		logger.info(f"Pixel classifier saved to {filename}.")
		return filename

	@_iterable_idx(tqdm_kw=dict(desc="Predicting probabilities", unit="image"))
	def pixel_classifier_predict_prob(self, idx=None, plot=False, **plot_kwargs):
		basename = self.basenames[idx]
		image = self.get_data(idx=idx, data_type="image")

		prob = self.pixel_classifier.predict_prob(images=image)

		# save probabilities to file

		filename = os.path.join(self.dir_root, DATA_TYPES.prob.dirname,
				basename + DATA_TYPES.prob.ext)
		self._pickle_dump(prob, filename)
		logger.debug(f"Probabilities saved to {filename}.")

		# update paths
		self.paths[basename].prob = filename

		# update cache
		if basename in self.cache:
			data = self[basename]
			data.prob = prob
			self.cache.update(name=basename, item=data)

		if plot:
			self.plot_image_classification(idx, **plot_kwargs)

		return prob

	def pixel_classifier_predict(self, idx=None, plot=False, **plot_kwargs):
		"""
		Predict the most likely class for each pixel in the image.

		Parameters
		----------
		idx : int, str, slice, Iterable[int], Iterable[str]
			Index of the image to predict.
		plot : bool, optional
			If True, plot the predictions.
		**plot_kwargs : dict, optional
			Additional keyword arguments for plotting.

		Returns
		-------
		prob : ndarray
			Predicted probabilities for each pixel in the image.
		"""
		prob = self.pixel_classifier_predict_prob(idx=idx, plot=plot, **plot_kwargs)
		return np.argmax(prob, axis=-1) + 1

	@_iterable_idx(tqdm_kw=dict(desc="Predicting model masks", unit="image"), batch_size=CPSAM_EVAL_KW.batch_size)
	def cpsam_mask(self, idx=None, plot=False, **plot_kwargs):
		basenames = self.get_data(idx=idx, data_type="basename")
		images = self.get_data(idx=idx, data_type="image")
		if isinstance(idx, int):  # same as batch_size==1
			basenames = [basenames]
			images = [images]

		if self.cpsam is None:
			time.sleep(0.1)
			print()
			logger.info("Loading cellpose model...")

			import torch

			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

			from cellpose import models

			self.cpsam = models.CellposeModel(gpu=torch.cuda.is_available(), device=device)
			logger.info("Loaded Cellpose model.")

			logger.info(f"Using device: {device}")

			# import ray
			# ray.init(address='127.0.0.1:6379')
			pass

		if not self._is_tqdm_running:
			logger.info("Predicting model mask...")

		cpsam_outs = self.cpsam.eval(x=images, **CPSAM_EVAL_KW)  # (mask, flow, style)
		# Unfortunately, batching is done in each image separately, therefore there is no speedup in
		# using batch_size > 1. DataManger is ready for batching, but the model is not.

		if not self._is_tqdm_running:
			logger.info("Model mask predicted.")

		for basename, (mask, flow, style) in zip(basenames, (cpsam_outs[0], cpsam_outs[1], cpsam_outs[2])):
			cpsam_out = CPSAMEvalOut(mask=mask, flow=flow, style=style)

			# save mask to file
			filename = os.path.join(self.dir_root, DATA_TYPES.cpsam_out.dirname,
					basename + DATA_TYPES.cpsam_out.ext)
			self._pickle_dump(cpsam_out, filename)
			logger.debug(f"Model output saved to {filename}.")

			# update paths
			self.paths[basename].cpsam_out = filename

			# update cache
			if basename in self.cache:
				data = self[basename]
				data.cpsam_out = cpsam_out
				self.cache.update(name=basename, item=data)

			if plot:
				self.plot_image_classification(idx, **plot_kwargs)

		return cpsam_outs

	@_iterable_idx(tqdm_kw=dict(desc="Plotting images", unit="image"))
	def plot_image(self, idx=None,
			save_fig=False, imshow_kw=None, **set_props_kw):
		"""
		Plot the image.

		Parameters
		----------
		idx :           int, str, slice, Iterable[int], Iterable[str]
			Index of the image to plot.
		axs :           Axes or list[Axes]
			Axes object to plot on. If given as a list, plots will be
			[image] (or a part, depending on len(axs)).
			If None, a new Axes object will be created.
		save_fig :      bool, optional
			If True, save the figure to a file.
		imshow_kw :     dict, optional
		**set_props_kw : dict, optional

		Returns
		-------

		"""
		if imshow_kw is None:
			imshow_kw = dict()

		basename = self.basenames[idx]
		image = self.get_data(idx=idx, data_type="image")

		Ax = gr.Axes()
		Ax.axs[0, 0].imshow(image, cmap="gray", **imshow_kw)

		set_props_kw = dict(
				sup_title=f"{basename}",
				show_fig=not save_fig,
				save_file_name=os.path.join(self.dir_root, "figs", "images", basename) if save_fig else False,
				close_fig=save_fig,
		) | set_props_kw_image | set_props_kw
		Ax.set_props(**set_props_kw)

	@_iterable_idx(tqdm_kw=dict(desc="Plotting", unit="image"))
	def plot_image_classification(self, idx=None,
			axs=None, save_fig=False, imshow_kw=None, **set_props_kw):
		"""
		Plot predictions.

		Parameters
		----------
		idx :           int, str, slice, Iterable[int], Iterable[str]
			Index of the image to plot.
		axs :           Axes or list[Axes]
			Axes object to plot on. If given as a list, plots will be
			[image, probabilities, predictions, image+probabilities] (or a part, depending on len(axs)).
			If None, a new Axes object will be created.
		save_fig :      bool, optional
			If True, save the figure to a file.
		imshow_kw :     dict, optional
		**set_props_kw : dict, optional

		Returns
		-------

		"""
		if imshow_kw is None:
			imshow_kw = dict()
		if "cmap" in imshow_kw:
			raise ValueError("`cmap` keyword argument is not supported. Define `cmap` in the __cfg__ file instead.")
		if "alpha" in imshow_kw:
			raise ValueError("`alpha` keyword argument is not supported. Define `alpha` in the __cfg__ file instead.")

		data = self[idx]
		basename, image, labels, prob, cpsam_out = data.basename, data.image, data.labels, data.prob, data.cpsam_out

		mask, flow, style = None, None, None
		if cpsam_out is not None:
			mask, flow, style = cpsam_out.mask, cpsam_out.flow, cpsam_out.style

		if axs is None:  # create new axes
			axs = gr.Axes(shape=(2, 4), figsize=(15, 8), grid_layout=[[(0, 2), (0, 2)]]).axs
			"""
			+-----------------+-----------------+-----------------+-----------------+
			|                 |  Probabilities  |   Predictions   |   Image+Prob    |
			|      Image      +-----------------+-----------------+-----------------+
			|                 |                 |       Mask      |   Image+Mask    |
			+-----------------+-----------------+-----------------+-----------------+
			"""
			axs = axs[axs != 0]

		elif isinstance(axs, gr.Axes):
			axs = axs.axs

		if not isinstance(axs, Iterable):
			raise TypeError(f"`axs` must be an Axes object or a list of Axes objects (given {type(axs)} instead).")

		axs_iter = iter(axs)

		def plot_image(ax, image, cmap="gray", **imshow_kw):
			if len(ax.images) == 0:
				ax.imshow(image, cmap=cmap, **imshow_kw)
			else:
				ax.images[0].set_data(image)

		def plot_probabilities(ax, prob, cmap, **imshow_kw):
			X = np.einsum("...i,ij->...j", prob, np.array(cmap.colors))

			if len(ax.images) < 2:
				ax.imshow(X, cmap=cmap, **imshow_kw)
			else:
				ax.images[-1].set_data(X)

		def plot_predictions(ax, prob, cmap, **imshow_kw):
			X = np.argmax(prob, axis=-1) + 1

			if len(ax.images) < 2:
				ax.imshow(X, cmap=cmap, **imshow_kw)
			else:
				ax.images[-1].set_data(X)

		def plot_mask(ax, mask, cmap, **kwargs):
			mask = mask > 0  # convert to binary mask

			if len(ax.images) < 2:
				ax.imshow(mask, cmap=cmap, **kwargs)
			else:
				ax.images[-1].set_data(mask)

		# 0: Image
		ax = next(axs_iter)
		plot_image(ax, image, **imshow_kw)
		ax.set_title("Image")

		# 1: Image + Predictions
		ax = next(axs_iter)
		if prob is not None:
			plot_image(ax, image, **imshow_kw)
			plot_predictions(ax, prob, cmap=CMAP.rgba, **imshow_kw)
			ax.set_title("Image + Predictions")

		# 2: Predictions
		ax = next(axs_iter)
		if prob is not None:
			plot_predictions(ax, prob, cmap=CMAP.rgb, **imshow_kw)
			ax.set_title("Predictions")

			if ax.child_axes:  # colorbar exists
				pass
			else:  # add colorbar
				cax = ax.inset_axes(bounds=(0.01, 0.01, 0.03, 0.2))
				cax.grid(False)
				cbar = ax.figure.colorbar(mappable=ax.images[-1], cax=cax, orientation="vertical")
				cax.tick_params(axis="y", direction="in", color="none", pad=2)  # ticks
				cax.set_yticks(ticks=np.linspace(*cax.get_ylim(), 2 * len(LABELS) + 1)[1::2],
						labels=[f"{label_idx}: {label}" for (label, label_idx) in LABELS2IDX.items()],
						fontsize=7, rotation=0, color="white")  # tick labels

		# 3: Image + Mask
		ax = next(axs_iter)
		if mask is not None:
			plot_image(ax, image, **imshow_kw)
			plot_mask(ax, mask, cmap=CMAP.rgba_mask, **imshow_kw)
			ax.set_title("Image + Mask")

		# 4: Mask
		ax = next(axs_iter)
		if mask is not None:
			plot_mask(ax, mask, cmap=CMAP.rgb_mask, **imshow_kw)
			ax.set_title(f"Mask ({mask.max()} cells)")

		Ax = gr.Axes(axs=axs)
		set_props_kw = dict(
				sup_title=f"{basename}",
				show_fig=not save_fig,
				save_file_name=os.path.join(self.dir_root, "figs", "classification", basename) if save_fig else False,
				close_fig=save_fig,
		) | set_props_kw_image | set_props_kw
		Ax.set_props(**set_props_kw)

		data_instance = [axs[i].images for i in range(len(axs))]
		return data_instance

	def movie_image_classification(self, axs=None, save_file_name=None, **kwargs):
		"""
		Create a movie of the predictions.

		Parameters
		----------
		axs :           Axes or list[Axes]
			Axes object to plot on. If given as a list, plots will be
			[image, probabilities, predictions, image+probabilities] (or a part, depending on len(axs)).
			If None, a new Axes object will be created.
		save_file_name : str or None
			Path to save the movie. If None, the movie will not be saved.
		**kwargs :      sent to imshow

		Returns
		-------

		"""
		if save_file_name is None:
			save_file_name = os.path.join(self.dir_root, "figs", "classification_movie.gif")

		data_instance = self.plot_image_classification(idx=0, axs=axs, show_fig=False)
		axs = [data_instance[i]._axes for i in range(len(data_instance))]  # Get the Axes from the data instance

		def update_data(idx):
			"""Update the data for each axes."""
			return self.plot_image_classification(idx=idx, axs=axs, show_fig=False)

		logger.info(f"Creating movie with {len(self)} frames...")
		Ax = gr.Axes(axs=axs)
		Ax.plot_animation(axs=axs,
				func=update_data,
				n_frames=len(self),
				titles=self.basenames,
				**kwargs)
		Ax.save_fig(save_file_name)
		logger.info(f"Movie saved to {save_file_name}.")

	def plot_stats(self, axs=None, save_fig=False, **set_props_kw):
		"""
		Plot statistics of the pixel classifier.

		Parameters
		----------
		axs :           Axes or list[Axes]
			Axes object to plot on.
		save_fig :      bool, optional
			If True, save the figure to a file.
		**set_props_kw : dict, optional

		Returns
		-------

		"""
		if axs is None:
			axs = gr.Axes(shape=(2, 3), figsize=(15, 12)).axs.flatten()
		axs_iter = iter(axs)
		if not all(self.has_prob()):
			raise ValueError("Not all images have associated probabilities. Run `predict()` on all images first.")

		# Calculate statistics
		stats = Stats(
				count=np.zeros(len(self)),
				intensity=np.full((len(self), len(LABELS)), np.nan),
				avg_area=np.full(len(self), np.nan),
				avg_dist=np.full(len(self), np.nan),  # average distance between nuclei
		)

		logger.info("Calculating statistics...")
		tqdm_kw = TQDM_KW | dict(desc="Calculating statistics", unit="image")
		for idx in tqdm(range(len(self)), **tqdm_kw):
			data = self[idx]
			basename, image, labels, prob, cpsam_out = data.basename, data.image, data.labels, data.prob, data.cpsam_out
			pred = np.argmax(prob, axis=-1) + 1
			mask = cpsam_out.mask

			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
				warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars",
						category=RuntimeWarning)

				# Intensity
				for i, label_idx in enumerate(LABELS2IDX.values()):
					stats.intensity[idx, i] = np.mean(image[pred == label_idx])

				# Region properties
				props = regionprops(label_image=mask, intensity_image=image)

				# Count
				count = len(props)
				stats.count[idx] = count

				if count:  # if there are any detected nuclei
					# Average area
					stats.avg_area[idx] = np.mean([p.area for p in props])

					# Average distance
					centroids = [p.centroid for p in props]
					dist_mat = np.triu(distance_matrix(x=centroids, y=centroids), k=1)
					stats.avg_dist[idx] = np.mean(dist_mat[dist_mat > 0])  # average distance between nuclei

		logger.info("Finished calculating statistics.")

		def plot_xy(ax, x, y, title=None, xlabel=None, ylabel=None, ylim=None):
			y = np.asarray(y)
			if y.ndim == 1:
				ax.plot(x, y, color=CMAP.rgb.colors[LABELS2IDX.Nuclei - 1])  # Nuclei
			else:
				ax.plot(x, y)  # color cycler is set in `__cfg__.py`
			ax.set_title(title)
			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			ax.set_xlim(x[0], x[-1])
			ax.set_ylim(*ylim)

		x = range(len(self))

		# Nuclei Count vs. Time
		plot_xy(ax=next(axs_iter),
				x=x, y=stats.count,
				title="Nuclei Count",
				xlabel="Image Index",
				ylabel="Count",
				ylim=[0, None])

		# Avg. Nuclei Intensity vs. Time
		plot_xy(ax=next(axs_iter),
				x=x, y=stats.intensity,
				title="Avg. Nuclei Intensity",
				xlabel="Image Index",
				ylabel="Average Intensity",
				ylim=[0, None])

		# Avg. Nuclei Area vs. Time
		plot_xy(ax=next(axs_iter),
				x=x, y=stats.avg_area,
				title="Avg. Nuclei Area",
				xlabel="Image Index",
				ylabel="Area [pixels]",
				ylim=[0, None])

		# Average Nuclei Distance vs. Time
		plot_xy(ax=next(axs_iter),
				x=x, y=stats.avg_dist,
				title="Avg. Nuclei Distance",
				xlabel="Image Index",
				ylabel="Distance [pixels]",
				ylim=[0, None])

		# Nuclei Density vs. Time
		# raise NotImplementedError  # todo:

		Ax = gr.Axes(axs=axs)
		Ax.fig.legend(handles=axs[1].get_lines(), labels=LABELS, loc="upper right", ncol=len(LABELS), fontsize=10)
		set_props_kw = dict(
				sup_title=f"Pixel Classifier Statistics",
				show_fig=not save_fig,
				save_file_name=os.path.join(self.dir_root, "figs", "stats") if save_fig else False,
				close_fig=save_fig,
		) | set_props_kw
		Ax.set_props(**set_props_kw)
