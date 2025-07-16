import os
import time
import warnings
import random
import functools
from multiprocessing import Process
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable
from tqdm import tqdm
from natsort import natsorted
import numpy as np
from matplotlib.axes import Axes
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from scipy.spatial import distance_matrix
from prettytable import PrettyTable

from liron_utils.files import copy
from liron_utils.pure_python import dict_, NamedQueue, parallel_threading, tqdm_, is_debugger
from liron_utils.files import mkdirs
from liron_utils import graphics as gr

from __cfg__ import logger, AUTO_CONTRAST, set_props_kw_image, LABELS, CMAP, DATA_TYPES, Stats, \
    CPSAM_EVAL_KW, CPSAMEvalOut, get_tqdm_kw, CACHE_SIZE, DIR_OUTPUT, DEBUG
from utils import imread, imwrite, pickle_load, pickle_dump, ExcelData, get_image_paths
from utils_pixel_classifier import RandomForestPixelClassifier
from utils_napari import open_gui_for_segmentation
import tests


class DataManager:
    """
    Data manager for loading images and associated data.

    Parameters
    ----------
    dir_root :            str
        Path to the directory/tree containing images.
    sample_size :           int, float, or bool, optional
        Number of images to randomly sample from the directory.
        If given in the range (0, 1], it is interpreted as a fraction (True is the same as 1, i.e., use all data in
        random order. False will use all data in the order discovered by os.path.walk). If None, all images are used.
    labeled :               bool or None, optional
        If True, only labeled images will be loaded.
        If False, only unlabeled images will be loaded.
        If None, both labeled and unlabeled images will be loaded.
    path_excel :            str, optional
        Path to the Excel experiments file. Make sure to define columns correctly in EXCEL_COLUMNS
    random_forest_pixel_classifier :      PixelClassifier or str, optional
        Path to pixel classifier .pkl file or the object itself.
        If None, a new pixel classifier will be created.
    """

    def __init__(self, dir_root, *,
            labeled=None, sample_size=None, path_excel=None, date=None, pos=None,
            random_forest_pixel_classifier=None):

        tests.dir_exist(dir_root)
        self.dir_root = dir_root

        self.basenames, self.paths, self.counts = get_image_paths(dir_root=dir_root, labeled=labeled,
                sample_size=sample_size)
        self.num_samples = len(self.basenames)

        # Excel file
        if path_excel is None:
            self.time = dict_(
                    vector=range(self.num_samples),
                    limits=np.array([0, self.num_samples - 1]),
                    units="frames",
            )
            self._intensity_excel = None

        else:
            if date is None or pos is None:
                raise ValueError("If `path_excel` is given, both `date` and `pos` must be specified.")

            excel_data = ExcelData(path_excel, num_samples=self.num_samples, date=date, pos=pos)

            self.basenames = self.basenames[excel_data.min_frame:excel_data.max_frame + 1]
            self.paths = dict_(**{basename: self.paths[basename] for basename in self.basenames})
            self.num_samples = len(self.basenames)

            # Get time vector
            t0 = excel_data.time_after_cut[0]
            time_interval = excel_data.time_interval[0]
            self.time = dict_(
                    vector=(t0 + np.arange(self.num_samples) * time_interval) / 60,
                    limits=np.array([0, t0 + (self.num_samples - 1) * time_interval]) / 60,
                    units="hours",
            )

            # Get beta-catenin intensity
            lengths = [tf - ti + 1 for ti, tf in
                zip(excel_data.initial_frame_beta_catenin, excel_data.final_frame_beta_catenin)]
            self._intensity_excel = np.repeat(excel_data.beta_catenin_intensity, lengths)

            if len(self.time.vector) != len(self._intensity_excel):
                raise ValueError("Time vector must have same length as `intensity_excel`.")

        # Cache basename -> data
        self.cache = NamedQueue(max_size=CACHE_SIZE)

        # Pixel classifier
        self.pixel_classifier: RandomForestPixelClassifier = None
        if isinstance(random_forest_pixel_classifier, RandomForestPixelClassifier):
            self.pixel_classifier = random_forest_pixel_classifier
        elif isinstance(random_forest_pixel_classifier, str):
            self.pixel_classifier = RandomForestPixelClassifier().load_model(filename=random_forest_pixel_classifier)

        self._is_tqdm_running = False

    def __len__(self):
        return self.num_samples

    def __repr__(self):
        return (f"DataManager(#samples={self.num_samples}, "
                f"dir_root={self.dir_root})")

    def __getitem__(self, idx):
        """Get the image and associated data for a given index."""

        if isinstance(idx, (int, np.integer)):
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

    @functools.cached_property
    def _table_dir_tree(self):
        dirnames = natsorted(np.unique([os.path.dirname(basename) for basename in self.basenames]))

        # Determine max depth for table columns
        max_depth = max(max([len(Path(dirname).parts) for dirname in dirnames]), 1)

        field_names = [f"Level {i + 1}" for i in range(max_depth)] + list(DATA_TYPES.keys())

        table = PrettyTable(field_names=field_names)
        for i in range(max_depth):
            table.align[f"Level {i + 1}"] = "l"
        for col in DATA_TYPES:
            table.align[col] = "r"

        # Fill table
        last_paths = [None] * max_depth
        for dirname in dirnames:
            parts = Path(dirname).parts

            row = []
            for i in range(max_depth):
                val = parts[i] if i < len(parts) else ""

                # Show val only if any parent (up to level i) has changed
                if any((j >= len(parts) or last_paths[j] != parts[j]) for j in range(i + 1)):
                    row.append(val)
                else:
                    row.append("")

            # Update last_parts
            last_paths = list(parts) + [""] * (max_depth - len(parts))

            row += [self.counts[dirname].get(data_type, 0) for data_type in DATA_TYPES]

            table.add_row(row)

        # Add totals
        def get_column_widths(table: PrettyTable):
            col_widths = []
            for i, field in enumerate(table.field_names):
                max_len = len(str(field))
                for row in table.rows:
                    max_len = max(max_len, len(str(row[i])))
                col_widths.append(max_len)
            return col_widths

        totals = defaultdict(int)
        for dirname, counts in self.counts.items():
            for data_type, count in counts.items():
                totals[data_type] += count

        total_row = ["TOTAL"] + [""] * (max_depth - 1) + [
            f"{totals[data_type]} ({totals[data_type] / totals['image']:.2%})"
            if totals['image'] else totals[data_type] for data_type in DATA_TYPES]
        table.add_row(total_row)
        table.add_row(["-" * w for w in get_column_widths(table)])
        table.del_row(-2)
        table.add_row(total_row)
        return table

    def print_image_tree(self):
        """Recursively print image tree statistics with counts of images and associated data."""

        print(self._table_dir_tree)  # todo: fix total in case of excel data

    @staticmethod
    def _iterable_idx(tqdm_kw=None, use_threading=False, batch_size=1, shuffle=False):
        """
        Decorator for methods that iterate over indices of the dataset, allowing for batching and optional threading.

        Parameters
        ----------
        tqdm_kw :           dict, optional
            Keyword arguments for tqdm progress bar. If None, tqdm is disabled.
        use_threading :     bool, optional
            If True, the function will be executed in several threads.
        batch_size :        int, optional
            Size of the batch to send to the function at once. If 1, each index is processed individually.
            If batch_size > 1, the function will be called with a list of indices (useful for GPU processing).
        shuffle :           bool, optional
            If True, the indices will be shuffled before processing. If False, the indices will be processed in order.

        Returns
        -------

        """
        if tqdm_kw is None:
            tqdm_kw = dict(disable=True)
        tqdm_kw = get_tqdm_kw(**tqdm_kw)

        if use_threading and batch_size > 1:
            raise ValueError("Threading cannot be used with `batch_size`>1.")

        def decorator(func):

            @functools.wraps(func)
            def wrapper(self, idx=None, *args, **kwargs):

                if idx is None:
                    idx = list(range(self.num_samples))

                if isinstance(idx, (int, np.integer)):
                    idx = int(idx)
                    return func(self, idx, *args, **kwargs)

                elif isinstance(idx, str):
                    basename = idx
                    idx = self.paths[basename].idx
                    return func(self, idx, *args, **kwargs)

                elif isinstance(idx, (slice, Iterable)):
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
                    if self._is_tqdm_running:
                        tqdm_kw_["disable"] = True
                    if not tqdm_kw_["disable"]:
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

                    if not tqdm_kw_["disable"]:
                        self._is_tqdm_running = False
                        logger.info(f'Finished {tqdm_kw_["desc"].lower()} (total={len(out)}).', stacklevel=4)

                    return out

                else:
                    raise TypeError(f"Index must be an integer or a string (given {type(idx)} instead).")

            return wrapper

        return decorator

    # Data Manipulation
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

        if data_type == "idx":
            return idx
        elif data_type == "basename":
            return basename

        tests.data_type_valid(data_type)

        if data_type == "image":
            image = imread(os.path.join(self.dir_root, self.paths[basename].image), auto_contrast=AUTO_CONTRAST)
            return image

        elif data_type == "labels":
            if self.paths[basename].labels is None:
                return None
            labels = imread(os.path.join(self.dir_root, self.paths[basename].labels)).round().astype("uint8")
            return labels

        elif data_type == "prob":
            if self.paths[basename].prob is None:
                return None
            prob = pickle_load(os.path.join(self.dir_root, self.paths[basename].prob))
            return prob

        elif data_type == "cpsam_out":
            if self.paths[basename].cpsam_out is None:
                return None
            cpsam_out = pickle_load(os.path.join(self.dir_root, self.paths[basename].cpsam_out))
            return cpsam_out

        elif data_type == "figs":
            return None

        else:  # Unreachable because tests.data_type_valid(data_type) checks for valid data types
            raise ValueError

    @_iterable_idx()
    def has_data(self, idx=None, data_type=None):
        """
        Check if the image has associated data of a given type.

        Parameters
        ----------
        idx : int, str, slice, Iterable[int], Iterable[str]
            Index of the image to get.
        data_type : str
            Type of data to return.

        Returns
        -------

        Examples
        --------
        >>> dm = DataManager(dir_root="path/to/data")
        >>> dm.has_data(idx=0, data_type="image")
        """
        basename = self.basenames[idx]

        tests.data_type_valid(data_type)

        return getattr(self.paths[basename], data_type) is not None

    def get_labeled_data(self):
        """Get all images with associated labels."""
        return [self[idx] for idx in range(self.num_samples) if self.has_data(idx, data_type="labels")]

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

    @_iterable_idx()
    def get_dir_out(self, idx=None, data_type=None):
        basename = self.basenames[idx]
        os.path.join(self.dir_root, self.paths[basename][data_type])

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
            dir_target = os.path.join(self.dir_root, DATA_TYPES.figs.dirname, "_images")

        mkdirs(dir_target)

        basename = self.basenames[idx]
        image = self.get_data(idx=idx, data_type="image")
        imwrite(image=image, filename=os.path.join(dir_target, basename + DATA_TYPES.image.ext))

    # Random Forest Pixel Classifier
    def pixel_classifier_fit(self):
        idx = [i for i in range(self.num_samples) if self.has_data(i, data_type="labels")]

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
            filename = os.path.join(self.dir_root, DIR_OUTPUT, "pixel_classifier.pkl")
        return self.pixel_classifier.save_model(filename=filename)

    @_iterable_idx(tqdm_kw=dict(desc="Predicting probabilities", unit="image"))
    def pixel_classifier_predict_prob(self, idx=None, plot=False, **plot_frame_kw):
        basename = self.basenames[idx]
        image = self.get_data(idx=idx, data_type="image")

        prob = self.pixel_classifier.predict_prob(images=image)

        # save probabilities to file
        filename = os.path.join(DATA_TYPES.prob.dirname, basename + DATA_TYPES.prob.ext)
        pickle_dump(prob, os.path.join(self.dir_root, filename))
        logger.debug(f"Probabilities saved to {filename}.")

        # update paths
        self.paths[basename].prob = filename

        # update cache
        if basename in self.cache:
            data = self[basename]
            data.prob = prob
            self.cache.update(name=basename, item=data)

        if plot:
            self.plot_frame(idx, **plot_frame_kw)

        return prob

    def pixel_classifier_predict(self, idx=None, plot=False, **plot_frame_kw):
        """
        Predict the most likely class for each pixel in the image.

        Parameters
        ----------
        idx : int, str, slice, Iterable[int], Iterable[str]
            Index of the image to predict.
        plot : bool, optional
            If True, plot the predictions.
        **plot_frame_kw : dict, optional
            Additional keyword arguments for plotting.

        Returns
        -------
        prob : ndarray
            Predicted probabilities for each pixel in the image.
        """
        prob = self.pixel_classifier_predict_prob(idx=idx, plot=plot, **plot_frame_kw)
        return np.argmax(prob, axis=-1) + 1

    # Cellpose
    @functools.cached_property
    def cpsam_model(self):
        """
        Load the Cellpose model for segmentation.

        Returns
        -------
        cpsam : CellposeModel
            The loaded Cellpose model.
        """
        time.sleep(0.1)
        print()
        logger.info("Loading cellpose model...")

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        from cellpose import models

        cpsam_model = models.CellposeModel(gpu=torch.cuda.is_available(), device=device)
        logger.info("Loaded Cellpose model.")

        logger.info(f"Using device: {device}")

        # import ray
        # ray.init(address='127.0.0.1:6379')

        return cpsam_model

    @_iterable_idx(tqdm_kw=dict(desc="Predicting model masks", unit="image"), batch_size=CPSAM_EVAL_KW.batch_size)
    def cpsam_mask(self, idx=None, plot=False, **plot_frame_kw):
        basenames = self.get_data(idx=idx, data_type="basename")
        images = self.get_data(idx=idx, data_type="image")
        if isinstance(idx, (int, np.integer)):  # same as batch_size==1
            idx = [idx]
            basenames = [basenames]
            images = [images]

        cpsam_outs = self.get_data(idx=idx, data_type="cpsam_out")
        idx_none = [i for i, cpsam_out in enumerate(cpsam_outs) if cpsam_out is None]
        if len(idx_none) == 0:
            logger.debug(f"Images {idx} already have masks. Skipping.")

            if plot:
                self.plot_frame(idx, **plot_frame_kw)

            return cpsam_outs

        basenames = [basenames[i] for i in idx_none]
        images = [images[i] for i in idx_none]

        if not self._is_tqdm_running:
            logger.info("Predicting model mask...")

        cpsam_outs = self.cpsam_model.eval(x=images, **CPSAM_EVAL_KW)  # (mask, flow, style)
        # Unfortunately, batching is done in each image separately, therefore there is no speedup in
        # using batch_size > 1. DataManger is ready for batching, but the model is not.

        if not self._is_tqdm_running:
            logger.info("Model mask predicted.")

        for i, basename in enumerate(basenames):
            cpsam_out = CPSAMEvalOut(mask=cpsam_outs[0][i], flow=cpsam_outs[1][i], style=cpsam_outs[2][i])

            # save mask to file
            filename = os.path.join(DATA_TYPES.cpsam_out.dirname, basename + DATA_TYPES.cpsam_out.ext)
            pickle_dump(cpsam_out, os.path.join(self.dir_root, filename))
            logger.debug(f"Model output saved to {filename}.")

            # update paths
            self.paths[basename].cpsam_out = filename

            # update cache
            if basename in self.cache:
                data = self[basename]
                data.cpsam_out = cpsam_out
                self.cache.update(name=basename, item=data)

            if plot:
                self.plot_frame(basename, **plot_frame_kw)

        return cpsam_outs

    # Plot
    @_iterable_idx(tqdm_kw=dict(desc="Plotting images", unit="image"))
    def _plot_image(self, idx=None, save_fig=False, imshow_kw=None,
            **set_props_kw):  # todo: compare with _save_images and consider merging
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
                save_file_name=os.path.join(self.dir_root, DATA_TYPES.figs.dirname, "_images",
                        basename) if save_fig else False,
                close_fig=save_fig,
        ) | set_props_kw_image | set_props_kw
        Ax.set_props(**set_props_kw)

    @_iterable_idx(tqdm_kw=dict(desc="Plotting", unit="image"))
    def plot_frame(self, idx=None, which="default", axs=None, save_fig=False, imshow_kw=None, **set_props_kw):
        """
        Plot predictions.

        Parameters
        ----------
        idx :           int, str, slice, Iterable[int], Iterable[str]
            Index of the image to plot.
        which :         str or list[str] or list[list[str]], optional
            Which plots to create, given as a list of the following options with (possibly) plus signs to plot together:
            - "image": image
            - "prob": random forest pixel classifier nuclei probabilities
            - "pred": random forest pixel classifier nuclei predictions
            - "mask": Cellpose model cell masks
            - "mask_bounds": Cellpose model cell mask boundaries
            - "stats": experiment statistics (see `plot_stats`) with vertical line at the current image index (time)
            Predefined options are:
            - "default": ["image+mask_bounds", "mask+pred", "stats"]

            For example, if `which=["image", "image+mask_bounds", "prob", "mask+pred", "stats"]`,
            then len(which)==5 and the shape will be:
                1       2       3               4
            1	+---------------+---------------+---------------+
                │               │      <2>      │      <4>      │
            2	│      <1>      +---------------+---------------+
                │               │      <3>      │      <5>      │
                +---------------+---------------+---------------+
        axs :           Axes or list[Axes], optional
            Axes object to plot on. If given as a list, plots will be
            [image, probabilities, predictions, image+probabilities] (or a part, depending on len(axs)).
            If None, a new Axes object will be created.
            This argument is used for compatibility with other plotting functions.
        save_fig :      bool, optional
            If True, save the figure to a file and don't show it.
        imshow_kw :     dict, optional
        **set_props_kw : dict, optional

        Returns
        -------

        """
        basename = self.basenames[idx]

        # Check and parse inputs
        if save_fig:
            save_file_name = os.path.join(self.dir_root, DATA_TYPES.figs.dirname, "classification",
                    basename + DATA_TYPES.figs.ext)

            # Check if the file already exists
            if os.path.exists(save_file_name):
                logger.debug(f"Image {basename} already exists. Skipping.")
                return None
        else:
            save_file_name = False

        if imshow_kw is None:
            imshow_kw = dict()
        if "cmap" in imshow_kw:
            raise ValueError("`cmap` keyword argument is not supported. Define `cmap` in the __cfg__ file instead.")
        if "alpha" in imshow_kw:
            raise ValueError("`alpha` keyword argument is not supported. Define `alpha` in the __cfg__ file instead.")

        WHICH_VALUES = {
            "image",
            # Random Forest Pixel Classifier
            "prob",
            "pred",
            # Cellpose
            "mask",
            "mask_bounds",
            # Etc.
            "stats",
        }
        WHICH_PREDEFINED = dict(
                default=[["image", "mask_bounds"], ["mask", "pred"], ["stats"]],
        )

        if isinstance(which, str) and which in WHICH_PREDEFINED:
            which = WHICH_PREDEFINED[which]

        else:
            if isinstance(which, str):
                which = [which]

            if not isinstance(which, Iterable):
                raise TypeError(f"`which` must be a string or an iterable of strings (given {type(which)} instead).")

            def parse_which_str(which: str):
                """Parse a string of the form "<plot1>+<plot2>+..." into a list of strings."""
                which = which.replace(" ", "")

                if which in WHICH_VALUES:
                    which = [which]
                else:
                    which = which.split("+")
                    if not set(which) <= WHICH_VALUES:
                        raise ValueError(
                                f"Invalid `which` value '{which}'. Valid values are: {', '.join(WHICH_VALUES)}.")

                return which

            which = [parse_which_str(w) for w in which]

        if len(which) == 0:
            raise ValueError("`which` must contain at least one plot type.")
        elif len(which) in [1, 2]:
            shape = (1, len(which))
            # +---------------+---------------+
            # │               │               │
            # │      <1>      +      <2>      +
            # │               │               │
            # +---------------+---------------+
            grid_layout = None
        else:  # len(which) > 3
            shape = (2, 2 + len(which) // 2)
            # +---------------+---------------+---------------+
            # │               │      <2>      │      <4>      │
            # │      <1>      +---------------+---------------+
            # │               │      <3>      │      <5>      │
            # +---------------+---------------+---------------+
            grid_layout = [[(0, 2), (0, 2)]]

        axes_kw = dict(
                shape=shape,
                figsize=(15, 11),
                grid_layout=grid_layout,
        )

        if axs is None:  # create new axes
            axs = gr.Axes(**axes_kw).axs
            axs = axs[axs != 0]
        elif isinstance(axs, gr.Axes):
            axs = axs.axs
        if not isinstance(axs, Iterable):
            raise TypeError(f"`axs` must be an Axes object or a list of Axes objects (given {type(axs)} instead).")
        axs_iter = iter(axs)

        # Load data
        image = self.get_data(idx=idx, data_type="image")

        prob = None
        if any(["prob" in w or "pred" in w for w in which]):
            prob = self.get_data(idx=idx, data_type="prob")

        mask = None
        if any(["mask" in w or "mask_bounds" in w for w in which]):
            cpsam_out: CPSAMEvalOut = self.get_data(idx=idx, data_type="cpsam_out")
            if cpsam_out is not None:
                mask = cpsam_out.mask

        # Plot data
        def plot_image(ax, image, cmap="gray", **imshow_kw):
            if len(ax.images) == 0:
                ax.imshow(image, cmap=cmap, **imshow_kw)
            else:
                ax.images[0].set_data(image)

        def plot_probabilities(ax, prob, cmap, **imshow_kw):
            prob_color = np.einsum("...i,ij->...j", prob, np.array(cmap.colors))

            if len(ax.images) < 2:
                ax.imshow(prob_color, cmap=cmap, **imshow_kw)
            else:
                ax.images[-1].set_data(prob_color)

        def plot_predictions(ax, prob, cmap, **imshow_kw):
            pred = np.argmax(prob, axis=-1) + 1  # (0=unlabeled, 1=label1, 2=label2, ...)

            if len(ax.images) < 2:
                ax.imshow(pred, cmap=cmap, **imshow_kw)
            else:
                ax.images[-1].set_data(pred)

        def plot_mask(ax, mask, cmap, **kwargs):
            mask = mask > 0  # convert to binary mask (0=background, 1=nuclei)

            if len(ax.images) < 2:
                ax.imshow(mask, cmap=cmap, **kwargs)
            else:
                ax.images[-1].set_data(mask)

        for i, w in enumerate(which):
            ax = next(axs_iter)
            ax.set_title(" + ".join(w))

            cmap_prob = CMAP.rgb
            cmap_mask = CMAP.rgb_mask
            if len(w) > 1:
                cmap_prob = CMAP.rgba
                cmap_mask = CMAP.rgba_mask

            for plot in w:
                if plot == "image":
                    plot_image(ax, image, **imshow_kw)
                elif plot == "prob":
                    if prob is not None:
                        plot_probabilities(ax, prob, cmap=cmap_prob, **imshow_kw)
                elif plot == "pred":
                    if prob is not None:
                        plot_predictions(ax, prob, cmap=cmap_prob, **imshow_kw)
                        # if ax.child_axes:  # colorbar exists
                        # 	pass
                        # else:  # add colorbar
                        # 	cax = ax.inset_axes(bounds=(0.01, 0.01, 0.03, 0.2))
                        # 	cax.grid(False)
                        # 	cbar = ax.figure.colorbar(mappable=ax.images[-1], cax=cax, orientation="vertical")
                        # 	cax.tick_params(axis="y", direction="in", color="none", pad=2)  # ticks
                        # 	cax.set_yticks(ticks=np.linspace(*cax.get_ylim(), 2 * len(LABELS) + 1)[1::2],
                        # 			labels=[f"{label_idx}: {label}" for (label, label_idx) in LABELS2IDX.items()],
                        # 			fontsize=7, rotation=0, color="white")  # tick labels
                        pass
                elif plot == "mask":
                    if mask is not None:
                        plot_mask(ax, mask, cmap=cmap_mask, **imshow_kw)
                elif plot == "mask_bounds":
                    if mask is not None:
                        mask_bounds = find_boundaries(mask, mode='outer')
                        plot_mask(ax, mask_bounds, cmap=cmap_mask, **imshow_kw)
                elif plot == "stats":
                    self.plot_stats(idx_line=idx, ax=ax, save_fig=False, **set_props_kw)
                    axs[i] = None
                else:
                    raise ValueError(f"Invalid `which` value '{plot}'. Valid values are: {', '.join(WHICH_VALUES)}.")

        axs = [ax for ax in axs if ax is not None]
        Ax = gr.Axes(axs=axs)
        set_props_kw = dict(
                sup_title=f"{basename}",
                show_fig=not save_fig,
                save_file_name=save_file_name,
                close_fig=save_fig,
        ) | set_props_kw_image | set_props_kw
        Ax.set_props(**set_props_kw)

        data_instance = [axs[i].images for i in range(len(axs))]
        return data_instance

    def plot_movie(self, axs=None, save_file_name=None, plot_frame_kw=None, **plot_animation_kw):
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
        plot_frame_kw : dict, optional
        **plot_animation_kw :      sent to imshow

        Returns
        -------

        """
        if save_file_name is None:
            save_file_name = os.path.join(self.dir_root, DATA_TYPES.figs.dirname, "classification_movie.gif")
        if plot_frame_kw is None:
            plot_frame_kw = dict()

        # Plot once to get the data instance
        data_instance = self.plot_frame(idx=0, axs=axs, show_fig=False, **plot_frame_kw)
        axs = [data_instance[i]._axes for i in range(len(data_instance))]  # Get the Axes from the data instance

        logger.info(f"Creating movie with {len(self)} frames...")

        pbar = tqdm(total=len(self), **get_tqdm_kw(desc="Creating movie", unit="frame"))

        def update_data(idx):
            """Update the data for each axes."""
            data_instance = self.plot_frame(idx=idx, axs=axs, show_fig=False, **plot_frame_kw)
            pbar.update()
            return data_instance

        Ax = gr.Axes(axs=axs)
        Ax.plot_animation(axs=axs,
                func=update_data,
                n_frames=len(self),
                titles=self.basenames,
                **plot_animation_kw)
        Ax.save_fig(save_file_name)

        logger.info(f"Movie saved to {save_file_name}.")

    @functools.cached_property
    def stats(self):
        filename = os.path.join(self.dir_root, DIR_OUTPUT, "stats.pkl")
        if os.path.exists(filename):
            logger.info(f"Loading statistics from {filename}...")
            stats = pickle_load(filename)
            logger.info("Statistics loaded.")
            try:
                stats = Stats(**vars(stats))
                return Stats(**vars(stats))
            except TypeError as e:
                logger.error(f"Error loading statistics: {e}. Recalculating statistics.")

        # if not all(self.has_data(data_type="mask")):
        # 	raise ValueError("Not all images have masks. Run `cpsam_mask()` on all images first.")
        idx_cpsam_out = np.argwhere(self.has_data(data_type="cpsam_out")).flatten()

        # Calculate statistics
        stats = Stats(
                count=np.zeros(len(self), dtype="uint16"),  # number of detected nuclei
                avg_intensity=np.full(len(self), np.nan),  # average intensity
                avg_area=np.full(len(self), np.nan),  # average nuclei area
                sum_area_intensity=np.full(len(self), np.nan),  # sum of area * intensity
                avg_dist=np.full(len(self), np.nan),  # average distance between nuclei
                intensity_excel=self._intensity_excel,  # beta-catenin intensity from Excel
        )

        logger.info("Calculating statistics...")

        for idx in tqdm(idx_cpsam_out, **get_tqdm_kw(desc="Calculating statistics", unit="image")):
            image = self.get_data(idx=idx, data_type="image")
            cpsam_out: CPSAMEvalOut = self.get_data(idx=idx, data_type="cpsam_out")
            mask = cpsam_out.mask

            # Get statistics
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars",
                        category=RuntimeWarning)

                # Region properties
                props = regionprops(label_image=mask, intensity_image=image)

                # Count
                count = len(props)
                stats.count[idx] = count

                if count > 0:  # if there are any detected nuclei
                    # Avg. area
                    stats.avg_area[idx] = np.mean([p.area for p in props])

                    # Sum of area * intensity
                    stats.sum_area_intensity[idx] = np.sum([p.intensity_mean * p.area for p in props])

                    # Avg. Intensity
                    total_area = np.sum([p.area for p in props])
                    stats.avg_intensity[idx] = stats.sum_area_intensity[idx] / total_area

                    # Avg. distance between nuclei
                    centroids = [p.centroid for p in props]
                    dist_mat = np.triu(distance_matrix(x=centroids, y=centroids), k=1)
                    stats.avg_dist[idx] = np.mean(dist_mat[dist_mat > 0])

        logger.info("Finished calculating statistics.")

        # Save statistics to file
        logger.info(f"Saving statistics...")
        pickle_dump(stats, filename)
        logger.info(f"Statistics saved to {filename}.")

        return stats

    def plot_stats(self, idx_line=None, ax=None, save_fig=False, **set_props_kw):
        """
        Plot statistics of the pixel classifier.

        Parameters
        ----------
        idx_line :      int, str, optional
            Index of the image to plot a vertical line at. If None, no vertical line is plotted.
        ax :            Axes, optional
            Axes object to plot on.
        save_fig :      bool, optional
            If True, save the figure to a file.
        **set_props_kw : dict, optional

        Returns
        -------

        """

        # Check and parse inputs
        if save_fig:
            save_file_name = os.path.join(self.dir_root, DATA_TYPES.figs.dirname, "stats" + DATA_TYPES.figs.ext)

            # Check if the file already exists
            if os.path.exists(save_file_name):
                logger.debug(f"Statistics figure already exists. Skipping.")
                return None
        else:
            save_file_name = False

        set_props = False
        if ax is None:
            ax = gr.Axes(figsize=(15, 8)).axs[0, 0]
            set_props = True

        def plot_xy(ax, x, y, label=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None):
            if y.ndim == 1:
                ax.plot(x, y, label=label)  # Nuclei
            else:
                ax.plot(x, y)  # color cycler is set in `__cfg__.py`

            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

        t = self.time.vector

        class Quantity:
            def __init__(self, label, latex, y):
                self.label = label
                self.latex = latex
                self.y = y

        quantities_to_plot = [
            Quantity(label=r"Nuclei Count",
                    latex=r"$N_t$",
                    y=self.stats.count),
            Quantity(label=r"Avg. Intensity",
                    latex=r"$\frac{\sum_i {I_t^{(i)} \cdot A_t^{(i)}}}{\sum_i {A_t^{(i)}}}$",
                    y=self.stats.avg_intensity),
            Quantity(label=r"Sum Intensity $\times$ Area",
                    latex=r"$\sum_i {I_t^{(i)} \cdot A_t^{(i)}}$",
                    y=self.stats.sum_area_intensity),
            Quantity(label=r"Intensity (Excel)",
                    latex=r"",
                    y=self.stats.intensity_excel),
        ]

        for q in quantities_to_plot:
            q.y = np.asarray(q.y, dtype=float)
            q.y /= np.nanmax(q.y, axis=0)  # normalize to [0, 1]
            if len(q.y) != len(t):
                raise ValueError(f"Length of {q.label} ({len(q.y)}) does not match length of time vector ({len(t)}).")
            plot_xy(ax=ax, x=t, y=q.y, xlabel=f"Time [{self.time.units}]", xlim=self.time.limits)

        handles = ax.get_lines()
        labels = [f"{q.label} {q.latex}" for q in quantities_to_plot]
        ax.figure.legend(handles=handles, labels=labels, loc="lower left", ncol=len(labels) // 2 + 1, fontsize=10)

        ax.set_yticks([])

        if idx_line is not None:
            if isinstance(idx_line, str):
                idx_line = self.paths[idx_line].idx

            text = f"Frame: {idx_line}, Time: {t[idx_line]:.2f} {self.time.units}"
            for q in quantities_to_plot:
                if q.label == "Nuclei Count":
                    text += f"\n{q.label}: {int(q.y[idx_line])}"
                else:
                    text += f"\n{q.label}: {q.y[idx_line]:.1%}"

            ax.axvline(x=t[idx_line], color="black", linestyle="--", linewidth=1)
            textbox_kw = dict(
                    x=t[idx_line], y=0.95 * ax.get_ylim()[1],
                    s=text,
                    fontsize=7,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="Round, pad=0.2", facecolor="lightyellow", alpha=0.75)
            )
            textbox = ax.text(**textbox_kw)

            # Check if the text is outside the axes bounds and adjust its position
            renderer = ax.figure.canvas.get_renderer()
            bbox = textbox.get_window_extent(renderer=renderer)
            ax_bbox = ax.get_window_extent(renderer=renderer)
            if bbox.x1 > ax_bbox.x1:  # Remove previous text and add new one anchored from the right
                textbox.remove()
                ax.text(**(textbox_kw | dict(horizontalalignment="right")))

        if set_props:
            Ax = gr.Axes(axs=ax)
            # Ax.fig.legend(handles=axs[1].get_lines(), labels=LABELS, loc="upper right", ncol=len(LABELS), fontsize=10)
            set_props_kw = dict(
                    sup_title=f"Statistics",
                    show_fig=not save_fig,
                    save_file_name=save_file_name,
                    close_fig=save_fig,
            ) | set_props_kw
            Ax.set_props(**set_props_kw)

    def segment_in_napari(self, idx=None):
        basenames = self.get_data(idx=idx, data_type="basename")

        logger.info("Loading images...")
        images = np.array(self.get_data(idx=idx, data_type="image"))

        logger.info("Loading labels...")
        labels = self.get_data(idx=idx, data_type="labels")
        for i, label in enumerate(labels):
            if label is None:
                labels[i] = np.zeros(images[i].shape, dtype="uint8")
        labels = np.array(labels)

        logger.info("Loading masks...")
        cpsam_outs = self.get_data(idx=idx, data_type="cpsam_out")
        masks = np.zeros(images.shape, dtype="uint16")
        for i, cpsam_out in enumerate(cpsam_outs):
            if cpsam_out is not None:
                mask = cpsam_out.mask
                mask[mask > 0] = LABELS.nuclei.idx_napari
                masks[i] = mask

        open_gui_for_labeling_kw = dict(
                dir_root=self.dir_root,
                basenames=basenames,
                images=images,
                labels=labels,
                masks=masks,
        )
        if is_debugger and not DEBUG:  # open GUI in a separate process
            logger.info(f"Opening labeling GUI in a separate process...")
            Process(daemon=False, target=open_gui_for_segmentation, kwargs=open_gui_for_labeling_kw).start()
        else:
            logger.info(f"Opening labeling GUI...")
            open_gui_for_segmentation(**open_gui_for_labeling_kw)
