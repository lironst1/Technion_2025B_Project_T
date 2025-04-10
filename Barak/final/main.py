import argparse

from pathlib import Path
import pandas as pd
from data_statistics import single_directory_process
from utils import create_colored_ploar, create_polar_histogram


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=".",
        help="Directory containing the images",
    )

    parser.add_argument(
        "-r",
        "--frame_range",
        type=tuple_type,
        default=None,
        help="The range of frames to include in the video. Default is all frames.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="The output folder",
    )

    parser.add_argument(
        "-dl",
        "--directories_list",
        type=str,
        default=None,
        help="A file with a list of directories to process",
    )

    parser.add_argument(
        "--create_csv",
        action="store_true",
        help="Create new cells.csv with split_id",
    )

    parser.add_argument(
        "--create_clips",
        action="store_true",
        help="Create video clips of each split and full video",
    )

    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Store / Load from pickle",
    )

    parser.add_argument(
        "--polar",
        action="store_true",
        help="Generate polar plots of the cells around the split",
    )

    parser.add_argument(
        "--df_hist",
        action="store_true",
        help="Display histogram of the angle between the split dipole and the fibers",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.directories_list:
        with open(args.directories_list, "r") as f:
            directories = f.readlines()
            directories = [Path(directory.strip()) for directory in directories]

        directories_time_path = Path(args.directories_list).with_name(
            Path(args.directories_list).stem + "_times.csv"
        )
        directories_time = pd.read_csv(directories_time_path)
    else:
        directories = [Path(args.dir)]

    results = {}
    hists = [[], []]
    total_splits = 0
    for directory in directories:
        print(directory)
        frame_time = directories_time[directories_time["directory"] == str(directory)][
            "time"
        ].iloc[0]
        total_splits += single_directory_process(
            directory, frame_time, args, results, hists
        )
    print("Total splits:", total_splits)
    if args.polar or args.df_hist:
        graph_folder = Path(args.output) / "graphs"
        graph_folder.mkdir(parents=True, exist_ok=True)
    if args.polar:
        for key in results:
            result = results[key]
            create_colored_ploar(*result, folder_path=graph_folder / f"{key}")
    if args.df_hist:
        create_polar_histogram(hists[0], file_path=graph_folder / "fibers_angle.png")
        create_polar_histogram(hists[1], file_path=graph_folder / "main_axis_angle.png")


if __name__ == "__main__":
    # python main.py -p -d 2021_06_21_pos4 --create_clips
    main()
