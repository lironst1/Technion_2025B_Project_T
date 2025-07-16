import os
import time
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from utils import flatten_image_tree, get_image_paths
from utils_data_manager import DataManager

# from utils_napari import split_labels_tif


if __name__ == "__main__":
    PATH_DATA = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data"  # Path to the data directory
    get_path = lambda *args: os.path.join(PATH_DATA, *args)

    # flatten_image_tree(
    # 		dir_root=get_path("original"),
    # 		dir_target=get_path("2025_03_05", "data"),
    # 		date="2025_03_05",
    # 		pos=1,
    # 		# path_excel=get_path("betacatenin_head.xlsx"),
    # )
    # print_image_tree(get_path(r"original"))

    # dm = DataManager(dir_root=get_path("all_head"))
    # dm.pixel_classifier_fit()
    # file_pkl = dm.pixel_classifier_save()

    # file_pkl = get_path("all_head", "pixel_classifier.pkl")

    for date, pos in [("2025_03_05", 1)]:  # , ("2025_01_29", 1), ("2025_02_27", 1)]:
        print(f"Processing date: {date}, pos: {pos}")
        dm = DataManager(dir_root=get_path(rf"original\{date}\View{pos}"),
                path_excel=get_path("betacatenin_head.xlsx"),
                date=date, pos=pos
        )
        dm.segment_in_napari()
        # cpsam_out = dm.get_data(55, "cpsam_out")
        # if not all(dm.has_data(data_type="cpsam_out")):
        #     dm.cpsam_mask()
        # dm.plot_stats(save_fig=True)
        # dm.plot_frame(save_fig=True)

    # split_labels_tif(filename_labels=rel_path(r"train_2\labels.tif"),
    # 		dir_labeled_images=rel_path(r"train_2\data"),
    # 		dir_target=rel_path(r"all_head\labels")
    # )

    pass
