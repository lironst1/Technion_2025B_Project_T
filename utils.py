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

from liron_utils.files import copy
from liron_utils.files import open_file, mkdirs
from liron_utils.pure_python import dict_

from __cfg__ import logger, DEBUG, DIR_OUTPUT, IMAGE_EXTENSIONS, DATA_TYPES, get_tqdm_kw, EXCEL_COLUMNS, \
    AUTO_CONTRAST_KW, IGNORED_DIRS
import tests


def is_image(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".lnk":
        return is_image(filename.replace(".lnk", ""))
    if ext in IMAGE_EXTENSIONS:
        return True
    return False


def read_excel(path_excel):
    # tests.excel_permissions(path_excel)
    excel_data = pd.read_excel(path_excel, engine="openpyxl")
    return excel_data


def ignore_nan(x):
    return x[~np.isnan(x)]


class TimeVector:
    def __init__(self, num_samples: int, t0: float = 0, time_interval: float = 1,
            limits: list = None, units: str = "frames"):
        """
        Initialize a time vector.

        Parameters
        ----------
        num_samples : int
            Number of samples in the time vector.
        t0 : float, optional
            Initial time (default is 0).
        time_interval : float, optional
            Time interval between samples (default is 1).
        """

        self.t0 = t0
        self.time_interval = time_interval
        self.units = units

        if limits is None:
            self.limits = np.array([t0, t0 + (num_samples - 1) * time_interval])
        else:
            self.limits = np.array(limits)
            if len(self.limits) != 2:
                raise ValueError("Limits must be a list or array of two elements: [min, max].")
            if self.limits[0] > self.limits[1]:
                raise ValueError("Minimum limit must be less than or equal to maximum limit.")

        self.vector = np.arange(num_samples) * self.time_interval + self.t0

    def __repr__(self):
        return (f"TimeVector(num_samples={len(self.vector)}, "
                f"t0={self.vector[0]}, "
                f"time_interval={self.vector[1] - self.vector[0]:.3f}, "
                f"limits={self.limits[0]:.3f}-{self.limits[1]:.3f}, "
                f"units='{self.units}')")

    def __len__(self):
        return len(self.vector)


class ExcelData:
    """Container for Excel data."""

    def __init__(self, excel_data: str | pd.DataFrame, date: str | pd.Timestamp, pos: str | int):
        if isinstance(excel_data, pd.DataFrame):
            data = excel_data
        elif isinstance(excel_data, str):
            logger.debug(f"Loading Excel data from file '{excel_data}'...")
            data = read_excel(excel_data)
        else:
            raise ValueError(f"Invalid type for `excel_data`: {type(excel_data)}. "
                             f"Expected str (file path) or pd.DataFrame.")

        data[EXCEL_COLUMNS.date] = pd.to_datetime(data[EXCEL_COLUMNS.date], format="%d.%m.%y").ffill()
        data[EXCEL_COLUMNS.pos] = data[EXCEL_COLUMNS.pos].ffill()

        if isinstance(date, str):
            date = pd.to_datetime(date, format="%Y_%m_%d")
        if isinstance(pos, str):
            pos = int(pos.lower().replace("view", "").replace("pos", ""))

        data = data[(data[EXCEL_COLUMNS.date] == date) & (data[EXCEL_COLUMNS.pos] == pos)]
        if data.empty:
            raise ValueError(f"No matching rows found in Excel file for "
                             f"date {date.strftime('%Y_%m_%d')} and pos {pos}.")

        subset_dict = dict_(**{k: data[v].to_numpy() for k, v in EXCEL_COLUMNS.items()})

        self.date = subset_dict.date
        self.pos = subset_dict.pos.astype(int)
        self.time_after_cut = subset_dict.time_after_cut
        self.time_interval = subset_dict.time_interval
        self.main_orientation = subset_dict.main_orientation.astype(int)

        self.initial_frame_beta_catenin = ignore_nan(subset_dict.initial_frame_beta_catenin).astype(int)
        self.final_frame_beta_catenin = ignore_nan(subset_dict.final_frame_beta_catenin).astype(int)

        final_frame_beta_catenin = np.hstack(
                [self.initial_frame_beta_catenin[1:] - 1, self.max_frame])
        if not np.all(self.final_frame_beta_catenin == final_frame_beta_catenin):
            idx = np.where(self.final_frame_beta_catenin != final_frame_beta_catenin)[0][0]
            logger.warning(f"Excel data has inconsistent initial and final frames. Final frame at index {idx} "
                           f"is {self.final_frame_beta_catenin[idx]}, but next initial frame "
                           f"is {self.initial_frame_beta_catenin[1:][idx]}. Defining `final_frame=initial_frame[1:]-1`.")
            self.final_frame_beta_catenin = final_frame_beta_catenin

        self.beta_catenin_intensity = ignore_nan(subset_dict.beta_catenin_intensity).astype(int)

        if not len(self.initial_frame_beta_catenin) == len(self.final_frame_beta_catenin) == \
               len(self.beta_catenin_intensity):
            raise ValueError("Excel data columns 'initial_frame_beta_catenin', 'final_frame_beta_catenin', and "
                             "'beta_catenin_intensity' must have the same length.")

        if len(self.get_time_vector().vector) != len(self.beta_catenin_intensity_full):
            raise ValueError("Time vector must have same length as `intensity_excel`.")

    @property
    def min_frame(self):
        """Minimum frame number."""
        return self.initial_frame_beta_catenin[0]

    @property
    def max_frame(self):
        """Maximum frame number."""
        return self.final_frame_beta_catenin[-1]

    @property
    def num_samples(self):
        return self.final_frame_beta_catenin[-1] - self.initial_frame_beta_catenin[0] + 1

    def get_time_vector(self, frame_init: int = None, frame_final: int = None):
        if frame_init is None:
            frame_init = self.initial_frame_beta_catenin[0]
        if frame_final is None:
            frame_final = self.final_frame_beta_catenin[-1]
        time = TimeVector(
                num_samples=self.num_samples,
                t0=self.time_after_cut[0] / 60,  # [hours]
                time_interval=self.time_interval[0] / 60,  # [hours]
                limits=[frame_init * self.time_interval[0] / 60, frame_final * self.time_interval[0] / 60],  # [hours]
                units="hours",
        )
        return time

    @property
    def beta_catenin_intensity_full(self):
        repeats = [tf - ti + 1 for ti, tf in zip(self.initial_frame_beta_catenin, self.final_frame_beta_catenin)]
        return np.repeat(self.beta_catenin_intensity, repeats=repeats)


def flatten_image_tree(dir_root, dir_target=None, path_excel=None, date=None, pos=None, sep="__", overwrite=False,
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
    pos :                int, optional
        Position number to filter images by. If provided, only images from this position will be copied.
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
        dir_cur_data_pos = int(dir_cur_list[1].lower().replace("view", "").replace("pos", ""))

        if date is not None:  # Filter based on date
            if isinstance(date, str):
                date = pd.to_datetime(date, format="%Y_%m_%d")
            if dir_cur_data_date != date:
                continue

        if pos is not None:  # Filter based on position
            if isinstance(pos, str):
                pos = int(pos.lower().replace("view", "").replace("pos", ""))
            if dir_cur_data_pos != pos:
                continue

        if excel_data is not None:  # Filter based on Excel data
            # Check if the current date and pos match any rows in the Excel data
            matching_rows = excel_data[excel_data[EXCEL_COLUMNS.date] == dir_cur_data_date]
            if matching_rows.empty:
                logger.warning(f"No matching rows found in Excel file for "
                               f"date {dir_cur_data_date.strftime('%Y_%m_%d')}.")
                continue

            matching_rows = matching_rows[matching_rows[EXCEL_COLUMNS.pos] == dir_cur_data_pos]
            if matching_rows.empty:
                logger.warning(f"No matching rows found in Excel file for "
                               f"date {dir_cur_data_date.strftime('%Y_%m_%d')} and pos {dir_cur_data_pos}.")
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

        out = image.astype(float) * alpha + beta
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
        return pickle.load(f, fix_imports=True, encoding="latin1")


def pickle_dump(obj, filename):
    """Dump an object to a pickle file."""
    mkdirs(os.path.dirname(filename))
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def get_image_paths(dir_root: str,
        labeled: bool = None,
        sample_size: float = None,
        excel_data: pd.DataFrame = None, date: str | pd.Timestamp = None, pos: str | int = None):
    if isinstance(date, str):
        date = pd.to_datetime(date, format="%Y_%m_%d")  # todo: add format to __cfg__.py
    if isinstance(pos, str):
        pos = int(pos.lower().replace("view", "").replace("pos", ""))

    # Collect all files once
    def get_all_relative_files(dir_root: str, ignore_dirs: bool = False, date: pd.Timestamp = None, pos: int = None):
        """Recursively collects all file paths relative to root_dir."""

        if date is not None:
            date = date.strftime('%Y_%m_%d')
            pos = f"View{pos}"

        num_frames_before_filtering = None
        all_files = set()
        for current_root, dirs, files in os.walk(dir_root):
            if ignore_dirs:
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            rel_root = os.path.relpath(current_root, dir_root)
            if rel_root == ".":
                rel_root = ""

            if date is not None:
                if date in dirs:
                    dirs[:] = [d for d in dirs if d == date]
                    dir_root = os.path.join(dir_root, rel_root, date)
            if pos is not None:
                if pos in dirs:
                    dirs[:] = [d for d in dirs if d == pos]
                    dir_root = os.path.join(dir_root, rel_root, pos)
                if not (date in current_root and pos in current_root):
                    continue

            files = natsorted(files)
            if excel_data is not None:
                num_frames_before_filtering = len(files)
                files = files[excel_data.min_frame:excel_data.max_frame + 1]

            for filename in files:
                all_files.add(os.path.join(rel_root, filename))
        return all_files, dir_root, num_frames_before_filtering

    # Collect image files
    all_files, dir_root, num_frames_before_filtering = get_all_relative_files(dir_root=dir_root, ignore_dirs=True,
            date=date, pos=pos)
    image_paths = []
    for rel_path in all_files:
        parts = rel_path.split(os.sep)
        if is_image(parts[-1]):
            image_paths.append(os.path.join(parts[-3], parts[-2], parts[-1]) if len(parts) > 2 else rel_path)

    # Filter by labeled/unlabeled status
    all_files = get_all_relative_files(dir_root=dir_root, ignore_dirs=False, date=date, pos=pos)[0]
    if labeled is not None:
        filtered = []
        for img_path in image_paths:
            label_path = os.path.splitext(img_path)[0].replace(DATA_TYPES.image.dirname,
                    DATA_TYPES.labels.dirname) + DATA_TYPES.labels.ext
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

    return dir_root, basenames, paths, counts, num_frames_before_filtering
