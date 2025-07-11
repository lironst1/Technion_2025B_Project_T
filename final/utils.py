import os
import random
from collections import defaultdict
from natsort import natsorted
from prettytable import PrettyTable
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tifffile

from liron_utils.files import copy
from liron_utils.files import open_file, mkdirs
from liron_utils.pure_python import dict_

from __cfg__ import logger, IMAGE_EXTENSIONS, DATA_TYPES, get_tqdm_kw, EXCEL_COLUMNS, AUTO_CONTRAST_KW
import tests


def is_image(filename):
	ext = os.path.splitext(filename)[1].lower()
	if ext == ".lnk":
		return is_image(filename.replace(".lnk", ""))
	if ext in IMAGE_EXTENSIONS:
		return True
	return False


def read_excel(path_excel):
	tests.excel_permissions(path_excel)
	excel_data = pd.read_excel(path_excel)
	return excel_data


def ignore_nan(x):
	return x[~np.isnan(x)]


class ExcelData:
	"""Container for Excel data."""

	def __init__(self, path_excel, date=None, view=None):
		excel_data = read_excel(path_excel)

		excel_data[EXCEL_COLUMNS.date] = pd.to_datetime(excel_data[EXCEL_COLUMNS.date], format="%d.%m.%y").ffill()
		excel_data[EXCEL_COLUMNS.pos] = excel_data[EXCEL_COLUMNS.pos].ffill()

		# Filter frames by final frame of beta-catenin
		if isinstance(date, str):
			date = pd.to_datetime(date, format="%Y_%m_%d")
		if isinstance(view, str):
			view = int(view.lower().split("view")[-1])

		excel_data = excel_data[(excel_data[EXCEL_COLUMNS.date] == date) & (excel_data[EXCEL_COLUMNS.pos] == view)]
		if excel_data.empty:
			logger.warning(f"No matching rows found in Excel file for "
			               f"date {date.strftime('%Y_%m_%d')} and view {view}.")

		excel_data = dict_(**{k: excel_data[v].to_numpy() for k, v in EXCEL_COLUMNS.items()})

		self.date = excel_data.date
		self.pos = excel_data.pos.astype(int)
		self.time_after_cut = excel_data.time_after_cut
		self.time_interval = excel_data.time_interval
		self.main_orientation = excel_data.main_orientation.astype(int)
		self.initial_frame_beta_catenin = ignore_nan(excel_data.initial_frame_beta_catenin).astype(int)
		self.final_frame_beta_catenin = ignore_nan(excel_data.final_frame_beta_catenin).astype(int)
		self.beta_catenin_intensity = ignore_nan(excel_data.beta_catenin_intensity).astype(int)


def print_image_tree(dir_root):
	"""
    Print the directory tree under `dir_root`, showing the number of images and labels in each logical group.

    Parameters
    ----------
    dir_root :      str or Path
        Path to the parent directory containing 'data', 'labels' and 'prob' subdirectories.
    """
	dir_data = os.path.join(dir_root, DATA_TYPES.image.dirname)
	dir_labels = os.path.join(dir_root, DATA_TYPES.labels.dirname)

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


def flatten_image_tree(dir_root, dir_target=None, path_excel=None, date=None, view=None, sep="__", overwrite=False,
		symlink=True):
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
	sep :                 str, optional
		Separator used to join the directory names and image names in the target directory. Default is '__'.
	overwrite, symlink :    bool, optional
		See copy()

	Returns
	-------

	"""
	if dir_target is None:
		dir_target = dir_root

	excel_data = read_excel(path_excel) if path_excel is not None else None

	os.makedirs(dir_target, exist_ok=True)

	n_files = 0
	for dir_cur, _, filenames in tqdm(os.walk(dir_root, topdown=False),
			**get_tqdm_kw(desc="Copying files", unit="files")):
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
			matching_rows = excel_data[excel_data[EXCEL_COLUMNS.date] == dir_cur_data_date]
			if matching_rows.empty:
				logger.warning(f"No matching rows found in Excel file for "
				               f"date {dir_cur_data_date.strftime('%Y_%m_%d')}.")
				continue

			matching_rows = matching_rows[matching_rows[EXCEL_COLUMNS.pos] == dir_cur_data_view]
			if matching_rows.empty:
				logger.warning(f"No matching rows found in Excel file for "
				               f"date {dir_cur_data_date.strftime('%Y_%m_%d')} and view {dir_cur_data_view}.")
				continue

			max_frame = int(matching_rows[EXCEL_COLUMNS.final_frame_beta_catenin].max())
			filenames = filenames[:max_frame + 1]

		copy(src=[os.path.join(dir_cur, f) for f in filenames],
				dst=[os.path.join(dir_target, f"{rel_path.replace(os.sep, sep)}__{f}")
					for f in filenames],  # e.g., 2025_01_29__View1__Max_C1__image1.tif,
				overwrite=overwrite,
				symlink=symlink)

		n_files += len(filenames)

	logger.info(f"Finished copying {n_files} file{' links' if symlink else 's'} into {dir_target}.")
	open_file(dir_target)  # Open the target directory in File Explorer


def imread(filename, auto_contrast=False):
	"""Read an image from a file."""

	def fix_contrst(image, low_clip_percent, high_clip_percent):
		bit_depth = 8 * image.itemsize  # 16
		clip_value = 2 ** bit_depth - 1

		hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[2 ** bit_depth],
				ranges=[0, 2 ** bit_depth])
		accumulator = hist.cumsum()

		total = accumulator[-1]
		low_clip_value = total * low_clip_percent / 100.0
		high_clip_value = total * (1 - high_clip_percent / 100.0)

		minimum_gray = np.searchsorted(accumulator, low_clip_value)
		maximum_gray = np.searchsorted(accumulator, high_clip_value)

		# Avoid divide-by-zero
		if maximum_gray == minimum_gray:
			return image

		alpha = clip_value / (maximum_gray - minimum_gray)
		beta = -minimum_gray * alpha

		out = image.astype(np.float64) * alpha + beta
		out = np.clip(out, 0, clip_value).astype(f"uint{bit_depth}")
		return out

	image = tifffile.imread(filename)
	if auto_contrast:
		image = fix_contrst(image, **AUTO_CONTRAST_KW)

	return image


def imwrite(image, filename):
	"""Write an image to a file."""
	mkdirs(os.path.dirname(filename))

	ext = os.path.splitext(filename)[1].lower()
	if ext == ".tif":
		tifffile.imwrite(filename, image)
	else:
		raise ValueError(f"Unsupported file extension for writing: {ext}. Only .tif is supported.")


def pickle_load(filename):
	"""Load a pickle file."""
	with open(filename, "rb") as f:
		return pickle.load(f)


def pickle_dump(obj, filename):
	"""Dump an object to a pickle file."""
	mkdirs(os.path.dirname(filename))
	with open(filename, "wb") as f:
		pickle.dump(obj, f)


def get_image_basenames(dir_root, labeled=None, sample_size=None):
	# Discover image and data files
	filenames_images = os.listdir(os.path.join(dir_root, DATA_TYPES.image.dirname))
	filenames_images = natsorted([f for f in filenames_images if is_image(f)])

	if labeled is not None:  # Filter images based on labeled status
		filenames_labels = []
		for filename in filenames_images:
			filename_label = os.path.join(dir_root, DATA_TYPES.labels.dirname, filename)
			if labeled is True and not os.path.exists(filename_label) or \
					labeled is False and os.path.exists(filename_label):  # skip labeled/unlabeled images
				continue

			filenames_labels.append(filename_label)

		filenames_images = filenames_labels

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

		filenames_images = random.sample(population=filenames_images, k=min(sample_size, len(filenames_images)))
	else:
		filenames_images = natsorted(filenames_images)

	basenames: list[str] = [os.path.splitext(f)[0] for f in filenames_images]

	paths = dict_()  # {<basename>: {<idx>, <path_image>, <path_labels>, ...}}
	for idx, basename in enumerate(basenames):
		paths[basename] = dict_(idx=idx, **dict(zip(DATA_TYPES.keys(), len(DATA_TYPES) * [None])))
		for data_type, d in DATA_TYPES.items():
			path_data_type = os.path.join(dir_root, d.dirname, basename + d.ext)
			if os.path.exists(path_data_type):
				paths[basename][data_type] = path_data_type

	return basenames, paths
