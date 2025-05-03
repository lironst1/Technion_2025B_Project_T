import numpy as np
from cell_splits import CellSplit
from utils import (
	create_colored_ploar,
	create_polar_histogram,
	get_Q_tensor,
	project_shape_tensor,
)

import os
import pickle
from pathlib import Path

properties = ["area", "aspect_ratio", "perimeter"]
attributes = ["average_brightness", "circle_identify", "ellipse_identify"]
MAX_R = 5


def number_of_neighbors(cs, cell_id):
	"""
	Returns the number of neighbors for a given cell.

	Parameters:
	cs (CellSplit): An instance of the CellSplit class.
	cell_id (int): The ID of the cell.

	Returns:
	int: The number of neighbors for the given cell.
	"""
	graph = cs.get_graph_by_cell(cell_id)
	return len(list(graph.neighbors(cell_id)))


def Qrr_projection(cs, split, cell_id):
	"""
	Calculate the projection of the shape tensor Qrr onto the given axis.

	Parameters:
	cs (CellSplit): The CellSplit object containing the necessary data.
	split (Split): The Split object representing the cell split.
	cell_id (int): The ID of the cell.

	Returns:
	numpy.ndarray or None: The projection of the shape tensor Qrr onto the given axis, or None if Q is not available.
	"""
	angle = cs.get_orientation_from(cell_id, split.afters[0])
	axis = np.array([np.cos(angle), np.sin(angle)])
	Q = get_Q_tensor(cs, cell_id)
	if Q is None:
		return None
	return project_shape_tensor(Q, axis)


def Qrt_projection(cs, split, cell_id):
	"""
	Projects the shape tensor onto the axis defined by the orientation angle between two cells.

	Parameters:
	- cs: The cell split object.
	- split: The split object containing information about the split.
	- cell_id: The ID of the cell.

	Returns:
	- The projected shape tensor.

	Note:
	- If the shape tensor (Q) is None, None is returned.
	"""

	angle = cs.get_orientation_from(cell_id, split.afters[0])
	axis = np.array([np.sin(angle), -np.cos(angle)])
	Q = get_Q_tensor(cs, cell_id)
	if Q is None:
		return None
	return project_shape_tensor(Q, axis)


def single_directory_process(directory, frame_time, args, results, hists):
	"""
	Process a single directory containing cell images.

	Args:
		directory (Path): The directory containing the cell images.
		frame_time (float): The time interval between frames.
		args (argparse.Namespace): Command-line arguments.
		results (dict): Dictionary to store the results.
		hists (list): List to store the histograms.

	Returns:
		int: The number of splits found in the cell images.
	"""
	video_identifier = directory.name

	output_folder = Path(args.output) / video_identifier
	output_folder.mkdir(parents=True, exist_ok=True)

	if args.pickle:
		pickle_path = f"{video_identifier}.pickle"
		if os.path.isfile(pickle_path):
			with open(pickle_path, "rb") as f:
				cs = pickle.load(f)
		else:
			cs = CellSplit(directory, args.frame_range, output_folder)
			cs.find_splits()
			with open(pickle_path, "wb") as f:
				pickle.dump(cs, f)
	else:
		cs = CellSplit(directory, args.frame_range, output_folder)
		cs.find_splits()

	if cs.dm._image.shape != (1024, 1024):
		return

	print("Number of splits:", len(cs.splits))
	if len(cs.splits) == 0:
		return 0
	if args.create_csv:
		cs.add_splits_to_csv()

	if args.create_clips:
		cs.create_video()

	if args.polar or args.df_hist:
		graphs_output_folder = output_folder / "graphs"
		graphs_output_folder.mkdir(parents=True, exist_ok=True)
	if args.polar:
		r, t, v, time = cs.get_neighbers_values(number_of_neighbors, frame_time, MAX_R)
		create_colored_ploar(
				r, t, v, time, folder_path=graphs_output_folder / "neighbors"
		)
		results.setdefault("neighbors", [[], [], [], []])[0].extend(r)
		results["neighbors"][1].extend(t)
		results["neighbors"][2].extend(v)
		results["neighbors"][3].extend(time)

		r, t, v, time = cs.get_neighbers_values(
				Qrr_projection, frame_time, MAX_R, need_split=True
		)
		create_colored_ploar(
				r, t, v, time, folder_path=graphs_output_folder / "Qrr_projection"
		)
		results.setdefault("Qrr_projection", [[], [], [], []])[0].extend(r)
		results["Qrr_projection"][1].extend(t)
		results["Qrr_projection"][2].extend(v)
		results["Qrr_projection"][3].extend(time)

		r, t, v, time = cs.get_neighbers_values(
				Qrt_projection, frame_time, MAX_R, need_split=True
		)
		create_colored_ploar(
				r, t, v, time, folder_path=graphs_output_folder / "Qrt_projection"
		)
		results.setdefault("Qrt_projection", [[], [], [], []])[0].extend(r)
		results["Qrt_projection"][1].extend(t)
		results["Qrt_projection"][2].extend(v)
		results["Qrt_projection"][3].extend(time)

		for attribute in attributes:
			r, t, v, time = cs.get_neighbers_values(
					lambda cs, cell_id: cs.get_graph_by_cell(cell_id).nodes[cell_id][
						attribute
					],
					frame_time,
					MAX_R,
			)
			create_colored_ploar(
					r, t, v, time, folder_path=graphs_output_folder / f"{attribute}"
			)
			results.setdefault(attribute, [[], [], [], []])[0].extend(r)
			results[attribute][1].extend(t)
			results[attribute][2].extend(v)
			results[attribute][3].extend(time)

		for cell_property in properties:
			r, t, v, time = cs.get_neighbers_values(
					lambda cs, cell_id: cs.dm.cell_attribute(cell_id, cell_property),
					frame_time,
					MAX_R,
			)
			results.setdefault(cell_property, [[], [], [], []])[0].extend(r)
			results[cell_property][1].extend(t)
			results[cell_property][2].extend(v)
			results[cell_property][3].extend(time)
			create_colored_ploar(
					r, t, v, time, folder_path=graphs_output_folder / f"{cell_property}"
			)

	if args.df_hist:
		fibers_angle = cs.dipole_fibers_angle_histogram()
		main_axis_angle = cs.dipole_main_axis_angle_histogram()
		create_polar_histogram(
				fibers_angle, file_path=graphs_output_folder / "fibers_angle.png"
		)
		create_polar_histogram(
				main_axis_angle, file_path=graphs_output_folder / "main_axis_angle.png"
		)
		hists[0].extend(fibers_angle)
		hists[1].extend(main_axis_angle)
	return len(cs.splits)
