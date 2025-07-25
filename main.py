import warnings
import argparse
import numpy as np
import random
import pandas as pd

from liron_utils.pure_python import Logger

import __cfg__
from __cfg__ import logger
from utils import read_excel


def to_tuple(strings: str, dtype: type):
    if strings.startswith("(") or strings.endswith("["):
        strings = strings[1]
    if strings.endswith(")") or strings.endswith("]"):
        strings = strings[:-1]

    strings = [s.strip() for s in strings.split(",")]

    out = tuple(map(dtype, strings))
    return out


def to_tuple_int(strings: str):
    return to_tuple(strings, int)


def to_tuple_str(strings: str):
    return to_tuple(strings, str)


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
            "-d",
            "--dir",
            type=str,
            default="example",
            help=f"Path to a directory containing the images. "
                 f"Default is the current directory. "
                 f"Output will be saved in a subdirectory named '{__cfg__.DIR_OUTPUT}'. "
                 f"If output directory already exists, values will be read from it for processing (if they exist).",
    )

    parser.add_argument(
            "-p",
            "--print",
            action="store_true",
            default=False,
            help="Print image tree and exit.",
    )

    parser.add_argument(
            "-dl",
            "--directories_list",
            type=str,
            default=None,
            help="A .txt file with a list of directories to process. "
                 "All frames in each directory will be processed.",
    )

    parser.add_argument(
            "-e",
            "--excel",
            type=str,
            help=f"Path to an Excel file with experiment data. "
                 f"If provided, the script will read the following columns: {list(__cfg__.EXCEL_COLUMNS.keys())}. "
                 f"If `--date` and `--pos` are provided, the script will only process data in `--dir` based on these "
                 f"values. Otherwise, it will process all data in `--dir` (which should also appear in `--excel`).",
    )

    parser.add_argument(
            "--date",
            type=to_tuple_str,
            help=f"Date of the experiment in the format 'YYYY-MM-DD'. "
                 f"If provided, the script will filter the data in `--excel` based on this date. "
                 f"Multiple dates can be provided as a comma-separated list 'YYYY-MM-DD,YYYY-MM-DD'. "
                 f"Note that this will only work if `--excel` and `--pos` are provided.",
    )

    parser.add_argument(
            "--pos",
            type=to_tuple_int,
            help=f"View of the experiment, given as an integer. "
                 f"If provided, the script will filter the data in `--excel` based on these positions. "
                 f"Multiple positions can be provided as a comma-separated list '1,2,3'. "
                 f"Note that this will only work if `--excel` and `--date` are provided.",
    )

    parser.add_argument(
            "--view",
            type=to_tuple_int,
            help=f"An alias for `--pos`.",
    )

    parser.add_argument(
            "--sample_size",
            type=float,
            default=None,
            help="Number of images to randomly sample from the directory."
                 "If given in the range (0, 1], it is interpreted as a fraction "
                 "(True is the same as 1, i.e., use all data in random order. "
                 "False will use all data in the order discovered by os.path.walk). "
                 "If None, all images are used.",
    )

    parser.add_argument(
            "--labeled",
            action="store_true",
            default=False,
            help="Only process labeled images.",
    )

    parser.add_argument(
            "--unlabeled",
            action="store_true",
            default=False,
            help="Only process unlabeled images.",
    )

    parser.add_argument(
            "--no_cpsam_mask",
            action="store_true",
            default=False,
            help="Do not run Cellpose model (CPSAM) for segmentation. "
                 "By default, the script will run CPSAM for segmentation. "
                 "If set, the script will skip CPSAM segmentation and use existing masks (if available).",
    )

    parser.add_argument(
            "--no_plot",
            action="store_true",
            default=False,
            help="Do not plot and do not save figures. "
                 "By default, the script will generate plots and save them in the output directory."
                 "If set, no plots will be generated.",
    )

    parser.add_argument(
            "--plot_only_stats",
            action="store_true",
            default=False,
            help="Plot only statistics. "
                 "If set, the script will only generate statistics plot, without frame-by-frame plots.",
    )

    parser.add_argument(
            "--segment_manually",
            action="store_true",
            default=False,
            help="Segment images manually using napari. "
                 "If set, the script will open a napari viewer for manual segmentation.",
    )

    parser.add_argument(
            "-v",
            "--verbose",
            action=f"store_true",
            default=__cfg__.DEBUG,
            help="Enable verbose logging. "
                 "If set, debug messages will be printed to the logger and console.",
    )

    parser.add_argument(
            "--debug",
            action=f"store_true",
            default=__cfg__.DEBUG,
            help="Enable debug mode. "
                 "If set, debug messages will be printed to the logger and console, "
                 "and seed will be set to a fixed value for reproducibility.",
    )

    return parser.parse_args()


def main():
    r"""
    Main function to run the script.
    Parse command line arguments and execute the main logic.

    Examples
    --------
    python main.py --dir /path/to/images --output /path/to/output

    Returns
    -------

    """
    args = parse_args()

    __cfg__.DEBUG = args.debug

    # %% Logger
    if args.debug or args.verbose:
        logger.file_handler.setLevel(Logger.NAME2LEVEL.DEBUG)
        logger.console_handler.setLevel(Logger.NAME2LEVEL.DEBUG)

    # %% Debug Mode
    if args.debug:
        logger.info(f"Running in DEBUG mode with seed {__cfg__.SEED}. "
                    f"Warnings will be treated as errors.")
        np.random.seed(__cfg__.SEED)
        random.seed(__cfg__.SEED)
        warnings.filterwarnings("error")  # Treat warnings as errors for debugging

    # %% dir

    dir_roots = [args.dir]
    if args.dir is not None:  # single directory
        if args.directories_list is not None:
            raise ValueError("Cannot specify both `--dir` and `--directories_list`. "
                             "Please provide only one of these arguments.")
        logger.info(f"Processing directory '{args.dir}'.")

    else:  # directory list
        if args.directories_list is None:
            raise ValueError("Please provide either `--dir` or `--directories_list`.")

        logger.info(f"Reading directories list file '{args.directories_list}'.")
        with open(args.directories_list, "r") as f:
            dir_roots = f.readlines()  # todo: check

    # %% excel
    path_excel = args.excel
    dates = list(args.date) if args.date is not None else None
    if args.pos is not None and args.view is not None:
        raise ValueError("Cannot specify both `--pos` and `--view`. Please provide only one of these arguments.")
    elif args.view is not None:
        args.pos = args.view
    positions = list(args.pos) if args.pos is not None else None

    if path_excel is None:
        if dates is not None or positions is not None:
            raise ValueError("Please provide `--excel` argument when using `--date` or `--pos`.")
        excel_data = None
        dates = [None] * len(dir_roots)
        positions = [None] * len(dir_roots)

    else:
        if dates is None or positions is None:
            raise ValueError("Please provide both `--date` and `--pos` arguments when using `--excel`.")

        if args.directories_list is not None:  # TODO: extend in the future
            raise ValueError("Cannot specify dates with `--directories_list`. "
                             "Please use `--dir` with a multiple dates.")

        for i in range(len(dates)):
            dates[i] = pd.to_datetime(dates[i].replace("-", "_"), format="%Y_%m_%d")

        if len(dates) == 1:
            dates = dates * len(positions)  # repeat date for each pos
        if len(positions) == 1:
            positions = positions * len(dates)
        if len(dates) != len(positions):
            raise ValueError("Number of dates and positions must match. "
                             "Please provide the same number of dates and positions.")

        dir_roots = dir_roots * len(dates)

        logger.info(f"Loading Excel data from file '{path_excel}'...")
        excel_data = read_excel(path_excel)

        logger.info(
                f"""Processing dates: {" | ".join([f"{d.strftime('%Y_%m_%d')} , View{p}" for d, p in zip(dates, positions)])}.""")

    # %% Labeled
    if args.labeled and args.unlabeled:
        raise ValueError("Cannot specify both `--labeled` and `--unlabeled`. "
                         "Please provide only one of these arguments.")
    labeled = None
    if args.labeled:
        labeled = True
        logger.info("Processing labeled images.")
    if args.unlabeled:
        labeled = False
        logger.info("Processing unlabeled images.")

    # %% Sample Size
    sample_size = args.sample_size

    # %% Import all other files only AFTER updating `__cfg__` values
    from utils_data_manager import DataManager

    for dir_root, date, pos in zip(dir_roots, dates, positions):
        dm = DataManager(
                dir_root=dir_root.strip(),
                labeled=labeled,
                sample_size=sample_size,
                excel_data=excel_data,
                date=date,
                pos=pos,
                random_forest_pixel_classifier=None,
        )

        if args.print:
            logger.info(f"Printing image tree for directory '{dir_root}'.")
            dm.print_image_tree()
            return

        if args.segment_manually:
            dm.segment_in_napari()
            return

        if not args.no_cpsam_mask:
            dm.cpsam_mask()

        if not args.no_plot:
            dm.plot_stats(show_fig=False, save_fig=True)
            if not args.plot_only_stats:
                dm.plot_frame(show_fig=False, save_fig=True)


if __name__ == "__main__":
    main()
