import os
import random
from collections import defaultdict
from natsort import natsorted
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tifffile
from typing import Optional, Union

from liron_utils.files import copy
from liron_utils.files import open_file, mkdirs
from liron_utils.pure_python import dict_

from __cfg__ import logger, IMAGE_EXTENSIONS, DATA_TYPES, get_tqdm_kw, EXCEL_COLUMNS, AUTO_CONTRAST_KW, IGNORED_DIRS
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

    def __init__(self, path_excel, num_samples, date=None, view=None):
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

        self.min_frame = int(self.initial_frame_beta_catenin[0])
        self.max_frame = int(self.final_frame_beta_catenin[-1])

        # fix human errors in Excel file
        if self.max_frame == num_samples:
            logger.warning(f"Excel data has max_frame={self.max_frame}, which is equal to the number of samples "
                           f"({num_samples}). Changing it to {self.max_frame - 1}.")
            self.final_frame_beta_catenin[-1] = self.max_frame = num_samples - 1

        final_frame_beta_catenin = np.hstack([self.initial_frame_beta_catenin[1:] - 1, self.max_frame])
        if not np.all(final_frame_beta_catenin == self.final_frame_beta_catenin):
            idx = np.where(final_frame_beta_catenin != self.final_frame_beta_catenin)[0][0]
            logger.warning(f"Excel data has inconsistent initial and final frames. Final frame at index {idx} "
                           f"is {self.final_frame_beta_catenin[idx]}, but next initial frame "
                           f"is {self.initial_frame_beta_catenin[1:][idx]}. Defining `final_frame=initial_frame[1:]-1`.")
            self.final_frame_beta_catenin = final_frame_beta_catenin

        self.check_inputs()

    def check_inputs(self):
        if not all([
            len(self.initial_frame_beta_catenin) == len(self.final_frame_beta_catenin),
            len(self.initial_frame_beta_catenin) == len(self.beta_catenin_intensity),
            len(self.final_frame_beta_catenin) == len(self.beta_catenin_intensity),
        ]):
            raise ValueError("Excel data columns 'initial_frame_beta_catenin', 'final_frame_beta_catenin', and "
                             "'beta_catenin_intensity' must have the same length.")


# if not all((self.initial_frame_beta_catenin[1:] - self.final_frame_beta_catenin[:-1]) == 1):
# 	raise ValueError("Excel data columns 'initial_frame_beta_catenin' and 'final_frame_beta_catenin' must "
# 	                 "have consecutive frames (i.e., final frame of one row should be the initial frame of the "
# 	                 "next row).")


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


def get_all_relative_files(root_dir: str, ignore_dirs: bool = False) -> set[str]:
    """Recursively collects all file paths relative to root_dir."""
    all_files = set()
    for current_root, dirs, files in os.walk(root_dir):
        if ignore_dirs:
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        rel_root = os.path.relpath(current_root, root_dir)
        if rel_root == ".":
            rel_root = ""
        for filename in files:
            all_files.add(os.path.join(rel_root, filename))
    return all_files


def get_image_paths(
        dir_root: str,
        labeled: Optional[bool] = None,
        sample_size: Optional[Union[int, float, bool]] = None,
):
    image_dirname = DATA_TYPES.image.dirname
    label_dirname = DATA_TYPES.labels.dirname
    label_ext = DATA_TYPES.labels.ext

    # Collect all files once
    all_files = get_all_relative_files(dir_root, ignore_dirs=True)

    # Collect image files
    image_paths = []
    for rel_path in all_files:
        parts = rel_path.split(os.sep)
        if is_image(parts[-1]):
            image_paths.append(os.path.join(parts[-3], parts[-2], parts[-1]) if len(parts) > 2 else rel_path)

    all_files = get_all_relative_files(dir_root, ignore_dirs=False)

    # Filter by labeled/unlabeled status
    if labeled is not None:
        filtered = []
        for img_path in image_paths:
            label_path = os.path.splitext(img_path)[0].replace(image_dirname, label_dirname) + label_ext
            has_label = label_path in all_files
            if (labeled and has_label) or (not labeled and not has_label):
                filtered.append(img_path)
        image_paths = filtered

    # Sample if needed
    if sample_size is None or sample_size is False:
        image_paths = natsorted(image_paths)
    else:
        if sample_size is True:
            sample_size = len(image_paths)
        elif isinstance(sample_size, float) and 0 < sample_size <= 1:
            sample_size = int(sample_size * len(image_paths))
        elif isinstance(sample_size, int):
            if sample_size > len(image_paths):
                raise ValueError(f"Requested sample size {sample_size} exceeds total images {len(image_paths)}.")
        else:
            raise ValueError(f"Invalid sample_size: {sample_size}")
        image_paths = random.sample(image_paths, k=sample_size)

    # Collect metadata
    basenames = [os.path.splitext(p)[0] for p in image_paths]
    paths = dict_()
    counts = defaultdict(lambda: defaultdict(int))

    for idx, basename in enumerate(basenames):
        dirname, filename = os.path.split(basename)
        paths[basename] = dict_(idx=idx, **{k: None for k in DATA_TYPES})

        for data_type, dtype in DATA_TYPES.items():
            rel_path = os.path.join(dirname, dtype.dirname, filename + dtype.ext)
            if rel_path in all_files:
                paths[basename][data_type] = rel_path
                counts[dirname][data_type] += 1

    return basenames, paths, counts
