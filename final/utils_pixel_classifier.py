from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

from __cfg__ import logger, SIGMAS, RANDOM_FOREST_CLASSIFIER_KW, get_tqdm_kw
from utils import pickle_load, pickle_dump

import tests


class RandomForestPixelClassifier:
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

	def __init__(self, sigmas=SIGMAS, **random_forest_classifier_kw):
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
		for image, label in tqdm(zip(images, labels),
				**get_tqdm_kw(desc="Training PixelClassifier", unit="image", total=len(images))):
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

	def load_model(self, filename):
		"""Load a pixel classifier from a file."""
		tests.file_exist(filename)
		obj = pickle_load(filename)
		self.__dict__ = obj.__dict__
		logger.info(f"Pixel classifier loaded from {filename}.")
		return self

	def save_model(self, filename):
		"""Save the pixel classifier to a file."""
		if not filename.endswith(".pkl"):
			filename += ".pkl"
		pickle_dump(self, filename)
		logger.info(f"Pixel classifier saved to {filename}.")
		return filename
