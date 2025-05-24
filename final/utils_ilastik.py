import os
import subprocess
import numpy as np
import tempfile
from tqdm import tqdm
from skimage.filters.rank import windowed_histogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from __cfg__ import PATH_ILASTIK_EXE, IMAGE_EXTENSIONS, logger
import tests


def run_ilastik(path_project, dir_root, image_ext=".tif"):
	if not os.path.exists(path_project):
		raise ValueError(f"Project file not found at {path_project}")
	if not os.path.isdir(dir_root):
		raise ValueError(f"Path is not a directory: {dir_root}")
	if " " in path_project or " " in dir_root:  # Ilastik doesn't like spaces in the path_project
		tests.symlink_admin_priv()
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


class PixelClassifier:
	def __init__(self, sigmas=None, n_estimators=100, max_depth=None):
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
		n_estimators : int, optional
		    Number of trees in the Random Forest.
		max_depth : int or None, optional
		    Maximum depth of each tree in the Random Forest. None means unlimited depth.
		"""
		if sigmas is None:
			sigmas = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0]

		self.sigmas = sigmas
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth)
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
			size = int(2 * np.ceil(3 * sigma) + 1)
			local_var = windowed_histogram(image=image.astype(np.uint8), footprint=np.ones((size, size)))
			features.append(local_var)
			self.feature_names.append(f'local_variance_{sigma}')

		# Stack features into a (H, W, F) array
		feature_stack = np.stack(features, axis=-1)
		return feature_stack

	def fit(self, image, labels):
		"""
		Train the Random Forest classifier using labeled pixels in the image.

		Parameters
		----------
		image : np.ndarray
		    2D grayscale image.
		labels : np.ndarray
		    2D label mask of the same shape as `image`. Unlabeled pixels should have value 0.
		"""
		feature_stack = self._compute_features(image)
		X = feature_stack[labels > 0]
		y = labels[labels > 0]

		# Flatten features and scale
		X_flat = X.reshape(-1, X.shape[-1])
		X_scaled = self.scaler.fit_transform(X_flat)

		self.clf.fit(X_scaled, y)

	def predict_proba(self, image):
		"""
		Predict class probabilities for each pixel in the image.

		Parameters
		----------
		image : np.ndarray
		    2D grayscale image.

		Returns
		-------
		A 3D array of shape (H, W, C) where C is the number of classes.
		"""
		feature_stack = self._compute_features(image)
		H, W, N = feature_stack.shape
		X_flat = feature_stack.reshape(-1, N)
		X_scaled = self.scaler.transform(X_flat)

		proba = self.clf.predict_proba(X_scaled)
		# Reshape to (H, W, num_classes)
		proba_image = proba.reshape(H, W, -1)
		return proba_image

	def predict(self, image):
		"""
		Predict the most likely class for each pixel in the image.

		Parameters
		----------
		image : np.ndarray
		    2D grayscale image.

		Returns
		-------
		2D array of predicted class labels with shape (H, W).
		"""
		proba_image = self.predict_proba(image)
		prediction = np.argmax(proba_image, axis=-1) + 1  # Classes start from 1
		return prediction
