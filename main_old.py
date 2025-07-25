import os
import time
import numpy as np
import matplotlib.pyplot as plt

from liron_utils import graphics as gr

from __cfg__ import logger
from utils import flatten_image_tree, get_image_paths
from utils_data_manager import DataManager

# from utils_napari import split_labels_tif


if __name__ == "__main__":
    PATH_DATA = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data"  # Path to the data directory
    get_path = lambda *args: os.path.join(PATH_DATA, *args)

    # flatten_image_tree(
    # 		dir_root=get_path("original", "2025_03_05", "View1"),
    # 		dir_target=get_path("2025_03_05", "View1", "Max_C1"),
    # 		# date="2025_03_05",
    # 		# pos=1,
    # 		# path_excel=get_path("betacatenin_head.xlsx"),
    #         ignore_dirs=True,
    # )
    # print_image_tree(get_path(r"original"))

    # dm = DataManager(dir_root=get_path("all_head"))
    # dm.pixel_classifier_fit()
    # file_pkl = dm.pixel_classifier_save()
    # file_pkl = get_path("all_head", "pixel_classifier.pkl")
    dm = DataManager(dir_root=get_path(rf"original\2025_03_05\View1"),
            excel_data=get_path("betacatenin_head.xlsx"),
            date="2025_03_05", pos=1,
    )
    dm.plot_frame(10, save_fig=False)
    # dm.segment_in_napari()
    # cpsam_out = dm.get_data(55, "cpsam_out")
    # if not all(dm.has_data(data_type="cpsam_out")):
    #     dm.cpsam_mask()
    # dm.plot_stats(show_fig=True)
    # dm.plot_frame(save_fig=True)

    # split_labels_tif(filename_labels=rel_path(r"train_2\labels.tif"),
    # 		dir_labeled_images=rel_path(r"train_2\data"),
    # 		dir_target=rel_path(r"all_head\labels")
    # )

    # %% Run CPSAM on 3D data
    # dir = r"C:\Users\liron\OneDrive - Technion\Homework\2025B\114252 - Project T\Data\10.6.25_294"
    #
    # import tifffile
    #
    # x = tifffile.imread(os.path.join(dir, "TL1_fused_tp_0_ch_1.tif"))
    # logger.info(f"Image shape: {x.shape}")
    #
    # import torch
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
    #
    # from cellpose import models
    # from utils import pickle_dump
    #
    # cpsam_model = models.CellposeModel(gpu=False)
    # logger.info("Loaded Cellpose model.")
    #
    # logger.info(f"Using device: {device}")
    # cpsam_outs = cpsam_model.eval(x, batch_size=64, do_3D=True, stitch_threshold=0.1, z_axis=0)  # (mask, flow, style)
    # logger.info(f"Saving model output...")
    # pickle_dump(cpsam_outs, os.path.join(dir, "cpsam_out.pkl"))
    # logger.info(f"Model output saved.")
    pass
