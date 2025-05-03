import pandas as pd
from tifffile import imread
import re
import os
import numpy as np
import cv2


class DataManager:
	"""
	A class that manages data for cell splitting project.
	"""

	def __init__(self, parent_folder, output_folder):
		self.video_identifier = re.sub(r"(pos)(\d+)", r"\1_\2", parent_folder.name)
		if self.video_identifier.startswith("SD"):
			self.video_identifier = "_".join(self.video_identifier.split("_")[1:])
		if self.video_identifier.endswith("new") or self.video_identifier.endswith("old"):
			self.video_identifier = "_".join(self.video_identifier.split("_")[:-1])
		if self.video_identifier.endswith("top") or self.video_identifier.endswith("bottom"):
			self.video_identifier = "_".join(self.video_identifier.split("_")[:-1])
		self.parent_folder = parent_folder
		self.cells_folder = self.parent_folder / "Cells"
		self.segmentation_folder = self.cells_folder / "Segmentation"
		self._cells = None
		self._directed_bonds = None
		self._image_frame = -1
		self._segmentation_frame = -1
		self.output_folder = output_folder
		self._padding = -1

		self.video_frames_folder = self.output_folder / "video_frames"

		self.get_padding()

	def get_padding(self):
		"""
		Get the padding length for file names in the segmentation folder.
		"""
		for file_name in os.listdir(str(self.segmentation_folder)):

			if file_name.startswith(self.video_identifier):
				self._padding = len(file_name.split(".")[0].split("_")[-1])
				return

		temp_name = "_".join(self.video_identifier.split("_")[:-1])
		for file_name in os.listdir(str(self.segmentation_folder)):

			if file_name.startswith(temp_name):
				self._padding = len(file_name.split(".")[0].split("_")[-1])
				self.video_identifier = temp_name + "_" + file_name.split("_")[len(temp_name.split("_"))]
				return

	def cell_attribute(self, cell_id, attribute):
		"""
		Get the attribute value of a cell.

		Args:
			cell_id (int or tuple): The ID(s) of the cell(s).
			attribute (str): The attribute to retrieve.

		Returns:
			The attribute value(s) of the cell(s).
		"""
		if isinstance(cell_id, tuple):
			r = self.cells()[self.cells()["cell_id"].isin(cell_id)][attribute].values
		else:
			r = self.cells()[self.cells()["cell_id"] == cell_id][attribute].values[0]

		return r

	def image_for_video_file_path(self, frame_number):
		"""
		Get the file path for a video frame image.

		Args:
			frame_number (int): The frame number.

		Returns:
			The file path for the video frame image.
		"""
		return (
				self.video_frames_folder
				/ f"{self.video_identifier}_V_{frame_number:0{self._padding}d}.png"
		)

	def image_for_video(self, frame_number):
		"""
		Get the image for a video frame.

		Args:
			frame_number (int): The frame number.

		Returns:
			The image for the video frame.
		"""
		output_file = self.image_for_video_file_path(frame_number)
		if output_file.exists():
			image = cv2.imread(str(output_file))
		else:
			image = self.image(frame_number)
			image = np.stack([image // 256] * 3, axis=-1).astype(np.uint8)

		return image

	def delete_all_images_for_video(self):
		"""
		Delete all images for the video.
		"""
		if self.video_frames_folder.exists():
			for file_name in os.listdir(str(self.video_frames_folder)):
				os.remove(str(self.video_frames_folder / file_name))

		self.video_frames_folder.mkdir(parents=True, exist_ok=True)

	def save_image_for_video(self, image, frame_number):
		"""
		Save an image for a video frame.

		Args:
			image (ndarray): The image to save.
			frame_number (int): The frame number.
		"""
		output_file = self.image_for_video_file_path(frame_number)
		cv2.imwrite(str(output_file), image)

	def creat_video_writer(self, output_path, fps):
		"""
		Create a video writer object.

		Args:
			output_path (Path): The output file path.
			fps (float): The frames per second.

		Returns:
			The video writer object.
		"""
		height, width = self._image.shape
		return cv2.VideoWriter(
				str(output_path),
				cv2.VideoWriter_fourcc(*"XVID"),
				fps,
				(width, height),
				isColor=True,
		)

	def image(self, frame_number):
		"""
		Get the image for a frame.

		Args:
			frame_number (int): The frame number.

		Returns:
			The image for the frame.
		"""
		if frame_number != self._image_frame:
			self._image = imread(
					self.segmentation_folder
					/ f"{self.video_identifier}_T_{frame_number:0{self._padding}d}.tiff"
			)
			self._image_frame = frame_number
		return self._image

	def get_frame_range(self):
		"""
		Get the range of frame numbers.

		Returns:
			The minimum and maximum frame numbers.
		"""
		highest_frame_number = 0
		for file_name in os.listdir(self.segmentation_folder):
			frame_number = int(file_name.split("_T_")[1].split(".")[0])
			if frame_number > highest_frame_number:
				highest_frame_number = frame_number
		return 1, highest_frame_number

	def segmentation(self, frame_number):
		"""
		Get the segmentation for a frame.

		Args:
			frame_number (int): The frame number.

		Returns:
			The segmentation for the frame.
		"""
		if frame_number != self._segmentation_frame:
			self._segmentation = imread(
					self.segmentation_folder
					/ f"{self.video_identifier}_T_{frame_number:0{self._padding}d}"
					/ "handCorrection.tif"
			)

			if len(self._segmentation.shape) != 2:
				if self._segmentation.shape[0] == 3:
					self._segmentation = self._segmentation[0, :, :]
				else:
					self._segmentation = self._segmentation[:, :, 0]

			self._segmentation_frame = frame_number
		return self._segmentation

	def cells(self):
		"""
		Get the cells data.

		Returns:
			The cells data as a DataFrame.
		"""
		if self._cells is None:
			self._cells = pd.read_csv(self.cells_folder / "cells.csv")
		return self._cells

	def directed_bonds(self):
		"""
		Get the directed bonds data.

		Returns:
			The directed bonds data as a DataFrame.
		"""
		if self._directed_bonds is None:
			self._directed_bonds = pd.read_csv(self.cells_folder / "directed_bonds.csv")
		return self._directed_bonds
