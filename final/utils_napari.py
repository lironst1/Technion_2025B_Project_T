import os
import numpy as np
import tifffile
from natsort import natsorted
from tqdm import tqdm

from liron_utils.pure_python import print_in_color

from __cfg__ import logger, DATA_TYPES, LABELS
from utils import is_image, imwrite


def split_labels_tif(filename_labels, dir_labeled_images, dir_target):
    """
    Split a 3D label TIF file into individual 2D slices and save them as separate TIF files.

    Parameters
    ----------
    filename_labels :       str
        Path to the 3D label TIF file of shape (N, H, W), where N is the number of slices.
    dir_labeled_images :    str
        Directory containing the original images (not their labeling).
    dir_target :            str
        Directory where the individual label slices will be saved.

    Returns
    -------

    """
    # Load the 3D label tif file
    labels_3d = tifffile.imread(filename_labels).astype("uint8")

    # Get a sorted list of image filenames in the original image directory
    filenames = os.listdir(dir_labeled_images)
    filenames = natsorted([f.replace(".lnk", "") for f in filenames if is_image(f)])

    # Check that the number of slices matches the number of images
    if len(filenames) != labels_3d.shape[0]:
        raise ValueError(f"Number of slices in labels file ({labels_3d.shape[0]}) "
                         f"does not match number of images ({len(filenames)}).")

    # Ensure output directory exists
    os.makedirs(dir_target, exist_ok=True)

    # Iterate over each slice
    for i, filename in tqdm(enumerate(filenames), total=len(filenames), desc="Processing slices"):
        label_slice = labels_3d[i]

        if np.any(label_slice):  # Save only if there are non-zero labels
            filename_out = os.path.join(dir_target, filename)
            if os.path.exists(filename_out):
                logger.warning(f"File {filename_out} already exists. Skipping.")
                continue
            tifffile.imwrite(filename_out, label_slice)

    logger.info("Label splitting completed.")


def open_gui_for_segmentation(dir_root, basenames, images, labels, masks):
    """
    Open a napari GUI for segmenting nuclei in images.

    Parameters
    ----------
    dir_root :      str
        Root directory containing the images and labels.
    basenames :     list[str]
        List of base filenames for the images.
    images :        np.ndarray
        Images to be displayed in napari.
    labels :        np.ndarray
        Labels corresponding to the images.
    masks :         np.ndarray
        Cellpose masks for the images.

    Returns
    -------

    """
    import napari
    from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QMessageBox, QShortcut
    from qtpy.QtGui import QKeySequence

    # %% Instructions
    label_names = [print_in_color(text=f"{k}={v.idx_napari}", foreground_rgb=v.color) for (k, v) in LABELS.items()]
    instructions = f"""
        Instructions:
        - Use the paint tool on the top left for segmentation.
        - Labels are: {", ".join(label_names)}.
        - Click 'Save Labels' or 'Ctrl+S' to save your work.
        - Click 'Done' when finished with this image.

        Tips:
        - Use different brush sizes for different nuclei.
        - Zoom in for precise labeling.
        - Use the eraser tool to correct mistakes.
        """
    print(instructions)

    # %% Open napari viewer
    viewer = napari.Viewer(title=f"Nuclei Segmentation - {os.path.basename(dir_root)}")

    # %% Add image layer
    images_layer = viewer.add_image(images, name="Images")

    # %% Add labels layer
    labels_layer = viewer.add_labels(np.copy(labels), name="Labels")
    labels_layer.brush_size = 10
    labels_layer.mode = "paint"

    def get_label_filename(basename):
        dirname, filename = os.path.split(basename)
        save_file_name = os.path.join(dirname, DATA_TYPES.labels.dirname, filename + DATA_TYPES.labels.ext)
        return os.path.join(dir_root, save_file_name)

    # %% Add masks layer
    masks_layer = viewer.add_labels(np.copy(masks), name="Cellpose Masks")
    masks_layer.brush_size = 10
    masks_layer.mode = "paint"

    # %% Create custom widget for saving
    widget = QWidget()
    layout = QVBoxLayout()

    # Save button
    save_button = QPushButton("Save Labels")
    save_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

    def is_saved(idx=None):
        """
        Check if the labels have been saved for the given indices.

        Parameters
        ----------
        idx :       int, list[int], optional
            Indices of the labels to check. If None, checks all labels.

        Returns
        -------
        bool
            True if the labels have been saved, False otherwise.
        """
        if idx is None:
            idx = range(len(labels))
        elif isinstance(idx, int):
            idx = [idx]

        current_labels = labels_layer.data

        for i in idx:
            label = labels[i]
            current_label = current_labels[i]

            if np.any(current_label != label):
                return False

        return True

    def save_button_callback():
        logger.debug("Save Labels button pressed.")
        try:
            current_labels = labels_layer.data

            for i, (basename, current_label) in enumerate(zip(basenames, current_labels)):
                if is_saved(idx=i):
                    logger.debug(f"Labels for {basename} are unchanged. Skipping save.")
                    continue  # Skip if labels are unchanged

                logger.debug(f"Saving labels for {basename}...")
                save_file_name = get_label_filename(basename)
                imwrite(current_label.astype("uint8"), save_file_name)

                logger.debug(f"Labels saved to {save_file_name}.")
                labels[i] = np.copy(current_label)  # Update the labels with the current labels

            # Show a confirmation message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Labels saved successfully!")
            msg.setWindowTitle("Save Confirmation")
            msg.exec_()

        except Exception as e:
            # Show an error message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Error saving labels: {str(e)}")
            msg.setWindowTitle("Save Error")
            msg.exec_()
            logger.error(f"Error saving labels: {e}")

    save_button.clicked.connect(save_button_callback)  # Connect button
    layout.addWidget(save_button)  # Add button to layout

    # Save shortcut (Ctrl+S)
    shortcut = QShortcut(QKeySequence("Ctrl+S"), viewer.window._qt_window)
    shortcut.activated.connect(save_button_callback)

    # Done button
    done_button = QPushButton("Done")
    done_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

    def done_buttun_callback(event=None):
        logger.debug(f"Done button pressed.")

        def close_window(close: bool):
            if close:
                logger.debug("Closing window...")
                viewer.close()
                if event is not None:
                    event.accept()

            else:
                logger.debug("Closing window cancelled.")
                if event is not None:
                    event.ignore()

        if is_saved():
            close_window(True)
            return

        else:  # Ask if user wants to save before closing
            reply = QMessageBox.question(widget,
                    'Unsaved Changes',
                    'You have unsaved changes. Do you want to save before closing?',
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

            if reply == QMessageBox.Yes:
                save_button_callback()
                if is_saved():  # Only close if save was successful
                    close_window(True)
                    return
            elif reply == QMessageBox.No:
                close_window(True)
                return

        close_window(False)

    done_button.clicked.connect(done_buttun_callback)  # Connect button
    layout.addWidget(done_button)  # Add button to layout

    widget.setLayout(layout)
    viewer.window.add_dock_widget(widget, area='right', name='Labeling Controls')  # Add widget to napari

    # %% Show the embedded IPython console
    console = viewer.window._qt_viewer.console  # napari-console plugin
    viewer.window.add_dock_widget(console, area='bottom', name='Console')
    console.push(dict(viewer=viewer, instructions=instructions, is_saved=is_saved, labels_layer=labels_layer))

    # %% Override close event to handle unsaved changes
    viewer.window._qt_window.closeEvent = done_buttun_callback

    # %% Show filename when switching images
    # Update status bar
    def update_filename_text(event):
        idx = int(event.value[0])
        viewer.status = f"Filename: {basenames[idx]}"

    viewer.dims.events.current_step.connect(update_filename_text)
    viewer.status = f"Filename: {basenames[viewer.dims.current_step[0]]}"  # Initialize the status bar with the first filename

    # Text Overlay
    points_layer = viewer.add_points(
            data=[[5, 5]],
            name="Filename Overlay",
            size=0,  # practically invisible dot
            opacity=0.8,  # fully transparent
            properties=dict(basename=[basenames[viewer.dims.current_step[0]]]),
            text=dict(
                    string="{basename}",
                    size=16,
                    color="white",
                    anchor="upper_left",
                    translation=[10, 10],
            )
    )

    def update_text_overlay(event):
        idx = int(event.value[0])
        points_layer.properties = dict(basename=[basenames[idx]])

    viewer.dims.events.current_step.connect(update_text_overlay)

    # %% Show the viewer and wait for the user to finish
    napari.run()  # viewer.show(block=True)
