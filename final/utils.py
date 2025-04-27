import os
import random
import shutil
from glob import glob
from multiprocessing import Pool
import subprocess
import numpy as np
import tempfile
import h5py
from tqdm import tqdm
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import skimage

from liron_utils.pure_python import NUM_CPUS, parallel_map, parallel_threading, dict_
from liron_utils.signal_processing import rescale
from liron_utils import graphics as gr

from __cfg__ import logger

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']

CMAP = dict_(
		rgb=ListedColormap([
			gr.hex2rgb(gr.COLORS.YELLOW),  # nuclei
			gr.hex2rgb(gr.COLORS.BLUE),  # background
			gr.hex2rgb(gr.COLORS.RED),  # dirt
		]),
		rgba=ListedColormap([
			gr.hex2rgb(gr.COLORS.YELLOW) + (0.25,),  # nuclei
			gr.hex2rgb(gr.COLORS.BLUE) + (0.2,),  # background
			gr.hex2rgb(gr.COLORS.RED) + (0.15,),  # dirt
		]),
)


def move_and_rename_files(dir_root, dir_target=None):
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
    │   ├── ABC_abc_image1.png
    │   ├── ABC_image1.png
    │   ├── ABC_image2.png
    │   ├── DEF_image1.png
    │   ├── DEF_image2.png


	Parameters
	----------
	dir_root :          str
		Path to the root directory containing the images.
	dir_target :        str
		Path to the target directory where the images will be moved. If not specified, dir_root is used.

	Returns
	-------

	"""
	if dir_target is None:
		dir_target = dir_root
	for dir_cur, _, filenames in os.walk(dir_root, topdown=False):
		for filename in filenames:
			rel_path = os.path.relpath(dir_cur, dir_root)
			filename_new = f"{rel_path.replace(os.sep, '_')}_{filename}"
			shutil.move(os.path.join(dir_cur, filename), os.path.join(dir_target, filename_new))
	logger.info("Finished moving files.")


def select_random_images(dir_root, dir_target=None, **kwargs):
	"""
	Select a subset of images from a directory and move them to separate directories for training and testing.

	Examples
	--------
	Select 100 training images and 50 testing images from dir_root to dir_target. This will assume the following
	directory structure:
	└── dir_root
    │   ├── image1.png
    │   ├── image2.png
    │   ├── ...

    └── dir_target
    │   ├── train
    │   │   ├── image1.png
    │   │   ├── ...
    │   ├── test
    │   │   ├── image2.png
    │   │   ├── ...
	>>> select_random_images(dir_root="~/Downloads/Data/all", dir_target="~/Downloads/Data", train=100, test=50)

	Parameters
	----------
	dir_root :      str
		Path to the root directory containing the images.
	dir_target :    str, optional
		Path to the target directory where the selected images will be copied. If not specified, dir_root is used.
	kwargs :        Keyword arguments for specifying the number of images for each set (training, testing, validation, etc.).

	Returns
	-------

	"""
	if len(kwargs) == 0:
		raise ValueError("At least one set must be specified (train, test, validation, etc.).")
	if dir_target is None:
		dir_target = dir_root

	sets = dict_(all=dict_(dir=dir_root, files=[]))
	# Filter the file list to only include image files
	sets.all.files = [f for f in os.listdir(sets.all.dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
	logger.info(f'Found {len(sets.all.files)} images in {sets.all.dir}.')

	num_desired_images = 0
	for key, n in kwargs.items():
		if n < 0:
			raise ValueError(f"n_{key} must be a non-negative integer (given {n} instead).")
		num_desired_images += n
		sets[key] = dict_(dir=os.path.join(dir_target, key), n=n, files=[])

	if num_desired_images > len(sets.all.files):
		raise ValueError("Total number of desired images exceeds the number of available images.")

	# Don't select images that are already in the train/test directories
	for key in sets:
		if key == "all":
			continue
		if os.path.exists(sets[key].dir):
			filenames_key = [f for f in os.listdir(sets[key].dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
			logger.info(f'Found {len(filenames_key)} images in {key}, which will not be considered for selection.')
			sets.all.files = [f for f in sets.all.files if f not in filenames_key]
			sets[key].n = max(sets[key].n - len(filenames_key), 0)

	# Randomly select images
	filenames_selected = random.sample(sets.all.files, num_desired_images)

	# Copy selected images
	num_images_cur = 0
	for key in sets:
		if key == "all":
			continue
		if sets[key].n == 0:
			continue

		sets[key].files = filenames_selected[num_images_cur:num_images_cur + sets[key].n]
		num_images_cur += sets[key].n

		# Copy selected images
		logger.info(f"Copying {sets[key].n} images to {key}...")
		os.makedirs(sets[key].dir, exist_ok=True)
		for f in sets[key].files:
			shutil.copyfile(src=os.path.join(sets.all.dir, f), dst=os.path.join(sets[key].dir, f))

	logger.info("Finished copying images.")


def run_ilastik_single_image(*args):
	ilastik_path, project_path, input_path, save_dir, save_format = args

	with tempfile.TemporaryDirectory() as temp_dir:
		output_file = os.path.join(temp_dir, "output.h5")

		command = [
			ilastik_path,
			f"--headless",
			f"--project={project_path}",
			f"--export_source=probabilities",
			f"--output_format=hdf5",
			f"--output_filename_format={output_file}",
			input_path
		]

		subprocess.run(command, check=True)

		with h5py.File(output_file, "r") as f:
			dataset_keys = list(f.keys())
			data = f[dataset_keys[0]][:]

	# Save output
	base_filename = os.path.splitext(os.path.basename(input_path))[0]
	if save_dir is not None:
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"{base_filename}.{save_format}")
		if save_format == "npy":
			np.save(save_path, data)
		elif save_format == "tif":
			tifffile.imwrite(save_path, data.astype(np.float32))  # or np.uint8 if needed
		else:
			raise ValueError(f"Unknown save_format: {save_format}")

	return data


def run_ilastik_on_folder_parallel(
		path_ilastik_exe,
		path_project,
		dir_root,
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
	path_ilastik_exe : str
	    Path to the Ilastik's 'run_ilastik.bat' script used for headless processing.
	path_project : str
	    Path to the Ilastik project file (.ilp) for loading the pre-trained model.
	dir_root : str
	    Path to the directory containing the images to process.
	file_ext : str, optional
	    File extension of image files to process (default is `"tif"`).
	n_workers : int, optional
	    Number of parallel processes to spawn for concurrent processing; defaults to using
	    all available CPUs minus one.
	dir_target : str, optional
	    Directory path for saving the output files. If not specified, results are not saved.
	save_format : str, optional
	    Format for saving output files, either `"npy"` or `"tif"` (default is `"npy"`).

	Returns
	-------
	list of np.ndarray
	    List of processed output arrays corresponding to the images in the specified folder.
	"""

	if not os.path.exists(path_ilastik_exe):
		raise ValueError(f"Ilastik executable not found at {path_ilastik_exe}")
	if not os.path.exists(path_project):
		raise ValueError(f"Project file not found at {path_project}")
	if not os.path.exists(dir_root):
		raise ValueError(f"Data path not found at {dir_root}")
	if save_format not in ["npy", "tif"]:
		raise ValueError(f"Invalid save format: {save_format}. Use 'npy' or 'tif'.")

	filenames = sorted([f for f in os.listdir(dir_root) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])

	if not filenames:
		raise ValueError(f"No images found in {dir_root}")
	logger.info(f"Found {len(filenames)} images. Running ilastik in parallel...")

	args_list = [(path_ilastik_exe, path_project, input_path, dir_target, save_format) for input_path in filenames]

	# outputs = parallel_threading(
	# 		func=run_ilastik_single_image,
	# 		iterable=args_list,
	# 		)
	outputs = []
	# with Pool(processes=n_workers) as pool:
	# 	for result in tqdm(pool.imap(run_ilastik_single_image, args_list), total=len(args_list)):
	# 		outputs.append(result)
	for args in tqdm(args_list):
		outputs.append(run_ilastik_single_image(*args))

	logger.info("Finished processing all images.")
	return outputs


def imread(filename):
	image = plt.imread(filename)
	image = skimage.exposure.equalize_adapthist(image, clip_limit=0.025)
	return image


def load_processed_data(dir_images, dir_probs=None, file_extension="npy"):
	"""
	Load all images and probabilities from a folder into a list.

	Parameters
	----------
	dir_images :              str
		Path to image folder. Probabilities are assumed to be in os.path.join(dir_images, "output").
	dir_probs :               str, optional
		Path to the folder containing the probabilities. If not specified, os.path.join(dir_images, "output") is used.
	file_extension :        str, optional
		Extension of saved files. Default is "npy"

	Returns
	-------
		list of np.ndarray
	"""
	if dir_probs is None:
		dir_probs = os.path.join(dir_images, "output")

	filenames_images = sorted([f for f in os.listdir(dir_images) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS])
	if not filenames_images:
		raise ValueError(f"No images found in {dir_images}.")

	filenames_probs = sorted(glob(os.path.join(dir_images, "output", f"*.{file_extension}")))
	if not filenames_probs:
		raise ValueError(f"No probabilities found in {dir_probs}.")

	if len(filenames_images) != len(filenames_probs):
		raise ValueError(
				f"Inconsistent number of images ({len(filenames_images)}) and probabilities ({len(filenames_probs)})")

	# Load images
	images = np.array([imread(os.path.join(dir_images, f)) for f in filenames_images])

	# Load processed files
	probs = np.array([np.load(f) for f in filenames_probs])

	logger.info(f"Loaded {len(filenames_images)} images and probabilities.")

	return images, probs


def plot_predictions(axs, im=None, prob=None, kind="probabilities", **kwargs):
	"""
	Plot predictions.

	Parameters
	----------
	axs :           Axes or list[Axes]
		Axes object to plot on. If given as a list, plots will be [image, probabilities, predictions] (or a part).
	im :          array_like, optional
		Image of shape (height, width).
	prob :          array_like, optional
		Array of shape (height, width, n_classes) containing the predicted probabilities.
	kind :          str
		Kind of plot to create:
		- "probabilities": Plot the predicted probabilities.
		- "predictions": Plot the predicted class labels.
		If len(Ax) == 1, 'im' and 'kind' will be plotted together.
		If len(Ax) == 2, 'im' and 'kind' will be plotted in Ax[0], Ax[1], respectively.
		If len(Ax) == 3, this parameter is ignored.
	**kwargs :      sent to imshow

	Returns
	-------

	"""
	if type(kind) is str:
		kind = [kind]
	if isinstance(axs, Axes):
		axs = [axs]
	elif len(axs) == 3:
		kind = ["probabilities", "predictions"]
	if im is None and prob is None:
		raise ValueError("Either 'im' or 'prob' must be provided.")

	cmap = CMAP.rgb

	if im is not None:
		im = np.array(im)
		axs[0].imshow(im, cmap="gray")
		if len(axs) == 1:
			cmap = CMAP.rgba

	if prob is not None:
		prob = np.array(prob)

		kwargs = dict(cmap=cmap) | kwargs

		if "probabilities" in kind:
			X = np.einsum("...i,ij->...j", prob, np.array(cmap.colors))
			axs[1].imshow(X, **kwargs)

		if "predictions" in kind:
			X = prob.argmax(axis=-1)
			axs[2].imshow(X, **kwargs)
