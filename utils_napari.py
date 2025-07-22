import os
import numpy as np
import tifffile
from natsort import natsorted
from tqdm import tqdm

from liron_utils.pure_python import print_in_color

from __cfg__ import logger, DATA_TYPES, LABELS, CMAP
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
    from napari.utils.notifications import notification_manager
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
    layer_images = viewer.add_image(images, name="Images")
    layer_images.editable = False

    # %% Add labels layer
    colormap = napari.utils.CyclicLabelColormap(np.vstack([[0, 0, 0], CMAP.rgb.colors]))
    layer_labels = viewer.add_labels(np.copy(masks), name="Labels",
            colormap=colormap)  # add masks as labels for faster segmentation
    layer_labels.editable = True
    layer_labels.brush_size = 10
    layer_labels.mode = "paint"
    layer_labels.selected_label = LABELS.nuclei.idx_napari  # Set default label to nuclei

    def cyclic_label(event):
        """ Cycle through the labels."""
        idx_napari = [label.idx_napari for label in LABELS.values()]
        idx_current = layer_labels.selected_label
        if idx_current not in idx_napari:  # Wrap around to the first label
            logger.debug(f"Selected label index {idx_current} not in LABELS. Wrapping to first/last label.")
            layer_labels.selected_label = idx_napari[idx_current % len(idx_napari) - 1]

    layer_labels.events.selected_label.connect(cyclic_label)

    # # %% Add masks layer
    # layer_masks = viewer.add_labels(np.copy(masks), name="Cellpose Masks", colormap=colormap)
    # layer_masks.editable = False

    # %% Filename layer (show filename both in the status bar and as an overlay on the image)
    viewer.status = f"Filename: {basenames[viewer.dims.current_step[0]]}"  # Initialize status bar

    layer_text = viewer.add_points(
            data=[[5, 5]],
            name="Filename Overlay",
            size=0,  # invisible dot
            opacity=0.8,
            properties=dict(basename=[basenames[viewer.dims.current_step[0]]]),
            text=dict(
                    string="{basename}",
                    size=14,
                    color="white",
                    anchor="upper_left",
                    translation=[5, 5],
            ))
    layer_text.editable = False

    def update_filename(event):
        idx = int(event.value[0])
        layer_text.properties = dict(basename=[basenames[idx]])  # Update text overlay with current filename
        viewer.status = f"Filename: {basenames[idx]}"  # Update status bar with current filename

    viewer.dims.events.current_step.connect(update_filename)

    # %% Create custom widget for saving
    save_box_widget = QWidget()
    save_box_layout = QVBoxLayout()

    # Save button
    button_save = QPushButton("Save Labels")
    button_save.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

    def get_label_filename(basename):
        dirname, filename = os.path.split(basename)
        save_file_name = os.path.join(dirname, DATA_TYPES.labels.dirname, filename + DATA_TYPES.labels.ext)
        return os.path.join(dir_root, save_file_name)

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

        current_labels = layer_labels.data

        for i in idx:
            label = labels[i]
            current_label = current_labels[i]

            if np.any(current_label != label):
                return False

        return True

    def save_button_callback():
        logger.debug("Save Labels button pressed.")
        try:
            current_labels = layer_labels.data

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

    button_save.clicked.connect(save_button_callback)  # Connect button
    save_box_layout.addWidget(button_save)  # Add button to layout

    # Save shortcut (Ctrl+S)
    shortcut = QShortcut(QKeySequence("Ctrl+S"), viewer.window._qt_window)
    shortcut.activated.connect(save_button_callback)

    # Done button
    button_done = QPushButton("Done")
    button_done.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

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
            reply = QMessageBox.question(save_box_widget,
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

    button_done.clicked.connect(done_buttun_callback)  # Connect button
    save_box_layout.addWidget(button_done)  # Add button to layout

    save_box_widget.setLayout(save_box_layout)
    viewer.window.add_dock_widget(save_box_widget, area='right', name='Labeling Controls')  # Add widget to napari

    # %% Show the embedded IPython console
    console = viewer.window._qt_viewer.console  # napari-console plugin
    viewer.window.add_dock_widget(console, area='bottom', name='Console')
    variables_to_send = dict(viewer=viewer, instructions=instructions)
    print(f"The following variables are available in the napari console: {', '.join(variables_to_send.keys())}.")
    console.push(variables_to_send)  # Add variables to console

    # %% Override close event to handle unsaved changes
    viewer.window._qt_window.closeEvent = done_buttun_callback

    # %% Set the active layer to labels
    viewer.layers.selection.active = layer_labels

    # %% Show the viewer
    napari.run(gui_exceptions=True)  # viewer.show(block=True)
