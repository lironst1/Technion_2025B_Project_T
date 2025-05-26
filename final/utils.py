import os
import random
from natsort import natsorted
from collections import defaultdict
from prettytable import PrettyTable
import pickle as pkl
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from collections.abc import Iterable
from skimage.exposure import equalize_adapthist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import windowed_histogram
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from liron_utils.files import copy
from liron_utils.pure_python import dict_, NamedQueue
from liron_utils.files import open_file, mkdirs
from liron_utils import graphics as gr

from __cfg__ import IMAGE_EXTENSIONS, PATH_DATA, get_path, HIST_EQUALIZE, SIGMAS, RANDOM_FOREST_CLASSIFIER_KW, \
	PATH_ILASTIK_EXE, set_props_kw_image, EQUALIZE_ADAPTHIST_KW, CMAP, logger
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
        Path to the parent directory containing 'data', 'labels' and 'output' subdirectories.
    """
	dir_data = os.path.join(dir_root, "data")
	dir_labels = os.path.join(dir_root, "labels")

	if not os.path.exists(dir_data) or not os.path.exists(dir_labels):
		raise FileNotFoundError(f"Expected 'data' and 'labels' directories inside {dir_root}")

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


def flatten_image_tree(dir_root, dir_target=None, path_excel=None, overwrite=False, symlink=True):
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
	dir_target :            str
		Path to the target directory where the images will be moved. If not specified, dir_root is used.
	path_excel :            str
		Path to an Excel file. If specified, the function will filter the images based on the data in the Excel file.
		The Excel file should contain columns: ["Date", "Pos", "final frame of beta catenin"]. When provided, the directory
		tree is assumed to have the following structure (<> is a placeholder for the actual values and their format):
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
	open_file(dir_target)  # Open the target directory in File Explorer


class PixelClassifier:
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
			X = features[label > 0]
			y = label[label > 0]
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

		probs = []
		for image in tqdm(images, desc="Predicting probabilities", unit="image", total=len(images)):
			features = self._compute_features(image)
			H, W, N = features.shape
			X_flat = features.reshape(-1, N)
			X_scaled = self.scaler.transform(X_flat)
			prob = self.clf.predict_proba(X_scaled)
			prob = prob.reshape(H, W, -1)  # todo: shape is incorrect
			probs.append(prob)

		return probs if len(probs) > 1 else probs[0]

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
		probs = self.predict_prob(images)
		if isinstance(probs, list):
			return [np.argmax(p, axis=-1) + 1 for p in probs]
		else:
			return np.argmax(probs, axis=-1) + 1


class DataManager:
	def __init__(self, dir_root, sample_size=None, cache_size=10, labeled=None, pixel_classifier=None):
		"""
		Data manager for loading images and associated data.
		
		Parameters
		----------
		dir_root :            str
			Path to the directory/tree containing images.
			Associated data is assumed to be in the same directory as each image, or in an ./output directory, e.g.:
		    └── all
			│   ├── data (links to images, flattened to a single directory)
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif.lnk
			│   │   ├── ...
			│   ├── labels
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1.tif
			│   │   ├── ...
			│   ├── output (pixel classifier outputs)
			│   │   ├── 2025_01_29__View1__Max_C1__1_beta_cat_25x_10min_T2_C1_Probabilities_.npy
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
		pixel_classifier :      PixelClassifier, optional
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

		# Load all associated data
		self.basenames = [os.path.splitext(f)[0] for f in filenames_images]
		self.paths = dict_()  # {<basename>: {<idx>, <path_image>, <path_asoc_data>}}
		for idx, filename in enumerate(filenames_images):
			basename = self.basenames[idx]

			path_image = os.path.join(dir_root, "data", filename)

			# Try default label and probability paths
			path_labels = os.path.join(dir_root, "labels", basename + ".tif")
			if not os.path.exists(path_labels):
				path_labels = None

			path_prob = os.path.join(dir_root, "output", basename + ".npy")
			if not os.path.exists(path_prob):
				path_prob = None

			self.paths[basename] = dict_(
					idx=idx,
					path_image=path_image,
					path_labels=path_labels,
					path_prob=path_prob,
			)

		# Cache basename -> data
		self.cache = NamedQueue(max_size=cache_size)

		# Pixel classifier
		if pixel_classifier is None:
			self.pixel_classifier: PixelClassifier = PixelClassifier()
			logger.info(f"Creating new pixel classifier.")
		elif isinstance(pixel_classifier, PixelClassifier):
			self.pixel_classifier = pixel_classifier
		elif isinstance(pixel_classifier, str):
			if not os.path.exists(pixel_classifier):
				raise ValueError(f"Pixel classifier file {pixel_classifier} does not exist.")
			with open(pixel_classifier, "rb") as file:
				self.pixel_classifier = pkl.load(file)
				logger.info(f"Loaded pixel classifier from {pixel_classifier}.")

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
		elif isinstance(idx, Iterable):
			return [self[i] for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		if idx < 0 or idx >= self.num_samples:
			raise IndexError(f"Index {idx} out of range (0-{self.num_samples - 1})")

		if basename in self.cache:  # Check if the image is already in memory
			data = self.cache[basename]
			image, labels, prob = data.image, data.labels, data.prob

			logger.debug(f"Image {idx}:{basename} loaded from cache. Cache size: {len(self.cache)}.")

		else:  # Load the image and associated data
			image = self.imread(self.paths[basename].path_image, hist_equalize=HIST_EQUALIZE)

			labels, prob = None, None
			if self.paths[basename].path_labels is not None:
				labels = self.imread(self.paths[basename].path_labels).round().astype("uint8")

			if self.paths[basename].path_prob is not None:
				prob = np.load(self.paths[basename].path_prob)

			data = dict_(image=image, labels=labels, prob=prob)
			self.cache.enqueue(name=basename, item=data)

			logger.debug(f"Image {idx}:{basename} loaded to cache. Cache size: {len(self.cache)}.")

		out = dict_(basename=basename, image=image, labels=labels, prob=prob)
		return out

	def get_labeled_data(self):
		"""Get all images with associated labels."""
		return [self[idx] for idx in range(self.num_samples) if self.has_labels(idx)]

	def has_labels(self, idx):
		"""Check if the image has labels."""
		if isinstance(idx, int):
			basename = self.basenames[idx]
		elif isinstance(idx, str):
			basename = idx
		elif isinstance(idx, slice):
			return [self.has_labels(i) for i in range(self.num_samples)[idx]]
		elif isinstance(idx, Iterable):
			return [self.has_labels(i) for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		return self.paths[basename].path_labels is not None

	def has_prob(self, idx):
		"""Check if the image has associated probabilities."""
		if isinstance(idx, int):
			basename = self.basenames[idx]
		elif isinstance(idx, str):
			basename = idx
		elif isinstance(idx, slice):
			return [self.has_prob(i) for i in range(self.num_samples)[idx]]
		elif isinstance(idx, Iterable):
			return [self.has_prob(i) for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		return self.paths[basename].path_prob is not None

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

		fields = ["path_image", "path_labels", "path_prob"]
		dirs = ["data", "labels", "output"]
		for field, dir in zip(fields, dirs):
			src = [os.path.join(self.dir_root, self.paths[basename][field]) for basename in self.paths if
				self.paths[basename][field] is not None]
			copy(src=src, dst=os.path.join(dir_target, dir))

	@staticmethod
	def imread(filename, hist_equalize=False):
		image = tifffile.imread(filename)
		if hist_equalize:
			image = equalize_adapthist(image, **EQUALIZE_ADAPTHIST_KW)
		return image

	def fit(self, idx=None):
		if idx is None:
			idx = [i for i in range(self.num_samples) if self.has_labels(i)]

		if isinstance(idx, int):
			pass
		elif isinstance(idx, str):
			basename = idx
			idx = self.paths[basename].idx
		elif isinstance(idx, slice):
			return [self.fit(i) for i in range(self.num_samples)[idx]]
		elif isinstance(idx, Iterable):
			return [self.fit(i) for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		if not self.has_labels(idx):
			raise IndexError(f"Index {idx} is not labeled.")

		data = self[idx]
		self.pixel_classifier.fit(images=data.image, labels=data.labels)
		return None

	def save_pixel_classifier(self, filename=None):
		"""Save the pixel classifier to a file."""
		if filename is None:
			filename = os.path.join(self.dir_root, "pixel_classifier.pkl")
		with open(filename, "wb") as f:
			pkl.dump(self.pixel_classifier, f)
		logger.info(f"Pixel classifier saved to {filename}.")
		return filename

	def predict(self, idx=None, plot=False, **plot_kwargs):
		if idx is None:
			idx = range(self.num_samples)
		if isinstance(idx, int):
			pass
		elif isinstance(idx, str):
			basename = idx
			idx = self.paths[basename].idx
		elif isinstance(idx, slice):
			return [self.predict(idx=i, plot=plot, **plot_kwargs) for i in range(self.num_samples)[idx]]
		elif isinstance(idx, Iterable):
			return [self.predict(idx=i, plot=plot, **plot_kwargs) for i in idx]
		else:
			raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

		basename = self.basenames[idx]
		data = self[idx]
		image = data.image
		data.prob = self.pixel_classifier.predict_prob(images=image)

		# save probabilities to file
		filename = os.path.join(self.dir_root, "output", basename + ".tif")
		mkdirs(os.path.dirname(filename))
		tifffile.imwrite(filename, data.prob)
		logger.debug(f"Probabilities saved to {filename}.")

		# update class
		self.paths[basename].path_prob = filename
		self.cache.update(name=basename, item=data)

		if plot:
			self.plot(idx, **plot_kwargs)

		return data.prob

	def plot(self, idx, axs=None, which="all", save_fig=False, **kwargs):
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
			raise ValueError("`cmap` keyword argument is not supported. Define `cmap` in the __cfg__ file instead.")

		data = self[idx]
		basename, image, labels, prob = data.basename, data.image, data.labels, data.prob

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
			raise ValueError(f"Invalid value in `which`. Allowed values are: {WHICH_VALUES} or 'all'.")

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
				ax.set_title("Image")
			else:
				if prob is None:
					logger.warning(f"Image {idx} has no associated data. Skipping {which[i]} plot.")
					continue

				if which[i] == "probabilities":
					plot_probabilities(ax, prob, cmap=CMAP.rgb, **kwargs)
					ax.set_title("Probabilities")
				elif which[i] == "predictions":
					plot_predictions(ax, prob, cmap=CMAP.rgb, **kwargs)
					ax.set_title("Predictions")
				else:  # which[i] == "image+probabilities":
					plot_image(ax, image, **kwargs)
					plot_probabilities(ax, prob, cmap=CMAP.rgba, **kwargs)
					ax.set_title("Image + Probabilities")

		Ax = gr.Axes(axs=axs)
		kwargs = dict(
				sup_title=f"{basename}",
				show_fig=not save_fig,
				save_file_name=os.path.join(self.dir_root, "figs", basename) if save_fig else False,
		) | set_props_kw_image
		Ax.set_props(**kwargs)

		return axs
