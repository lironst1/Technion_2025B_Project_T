import numpy as np
import networkx as nx
from itertools import combinations
from tqdm import tqdm

from split import Split
from data_manager import DataManager
from utils import (
    find_center_of_object,
    get_all_pixels_of_cell,
    almost_all,
)


class Const:
    """
    A class that defines constants used in the cell splitting process.
    """

    brightness_factor_before = 1.2
    brightness_factor_after = 1.1

    ellipse_identify_threshold = 1.15
    circle_identify_threshold = 1.2
    max_split_distace = 40
    max_split_area_diff = 350  # 1000
    max_after_split_area_diff = 600
    min_area_before_split = 1000
    time_window = 4

    @staticmethod
    def const_to_string():
        return "_".join(
            f"{key}_{value}"
            for key, value in Const.__dict__.items()
            if not key.startswith("__") and not callable(value)
        ).replace(".", "_")


close_functions = [
    lambda attr_1, attr_2: max(attr_1, attr_2) / min(attr_1, attr_2) > 1.2,
    lambda attr_1, attr_2: np.abs(attr_1 - attr_2) > 0.2,
    lambda attr_1, attr_2: np.abs(attr_1 - attr_2) > 0.2,
]


class CellSplit:
    def __init__(self, parent_folder, frame_range, output_folder):
        self.dm = DataManager(parent_folder, output_folder)
        self.output_folder = output_folder
        if frame_range is None:
            self.frame_range = self.dm.get_frame_range()
        else:
            self.frame_range = frame_range
        self.graphs = []

        self.before_split_log = []
        self.after_split_log = []
        self.splits = []
        self.centers = []
        self.find_centers()

    def create_video(self, fps=2):
        """
        Creates a video of the whole video with the split images.

        Args:
            fps (int, optional): Frames per second of the output video. Defaults to 2.
        """
        self.dm.delete_all_images_for_video()
        for split in self.splits:
            split.create_split_video()

        output_path = self.output_folder / f"{self.dm.video_identifier}_S.avi"
        video_writer = self.dm.creat_video_writer(output_path, fps)
        for frame_number in range(self.frame_range[0], self.frame_range[1] + 1):
            image = self.dm.image_for_video(frame_number)
            video_writer.write(image)
        video_writer.release()

    def add_splits_to_csv(self):
        """
        Adds split IDs to the cells in the DataFrame and saves the updated DataFrame to a CSV file.

        This method iterates over the splits and updates the split IDs for the cells in the DataFrame.
        It assigns the split ID to the cells that were split before and after the split event.
        The updated DataFrame is then saved to a CSV file named "cells_with_splits.csv" in the output folder.

        Returns:
            None
        """
        new_cells = self.dm.cells().copy()
        for i, split in enumerate(self.splits):
            for before in split.befores:
                new_cells.loc[new_cells["cell_id"] == before, "split_id"] = i + 1
            for after in split.afters:
                new_cells.loc[new_cells["cell_id"] == after[0], "split_id"] = i + 1
                new_cells.loc[new_cells["cell_id"] == after[1], "split_id"] = i + 1
        # split_id 0 everywhere else
        new_cells["split_id"] = new_cells["split_id"].fillna(0)
        new_cells.to_csv(self.output_folder / "cells_with_splits.csv")

    def get_neighbers_values(self, value_function, frame_time, max_r, need_split=False):
        """
        Get the neighbor values for each cell in the splits.

        Args:
            value_function (function): A function that calculates the value for each cell.
            frame_time (float): The time interval between frames.
            max_r (int): The maximum distance to consider for neighboring cells.
            need_split (bool, optional): Whether the split information is needed. Defaults to False.

        Returns:
            tuple: A tuple containing the distances, orientations, values, and times for each cell.

        """
        distances = []
        values = []
        orientations = []
        times = []

        for split in self.splits:
            dipole = split.dipole_xyz(0)
            dipole_orientation = np.arctan2(dipole[1], dipole[0])
            for t in split.time_range():
                split_cells = split.get_cells_with_distance_r(t, 0)
                if split_cells is None:
                    continue
                if t < 0:
                    split_cells = split_cells[0]
                for r in range(0, max_r + 1):
                    cells = split.get_cells_with_distance_r(t, r)
                    if cells is None:
                        continue
                    distances.extend([r] * len(cells))
                    orientations.extend(
                        [
                            self.get_orientation_from(cell, split_cells)
                            - dipole_orientation
                            for cell in cells
                        ]
                    )
                    if need_split:
                        values.extend(
                            [value_function(self, split, cell) for cell in cells]
                        )
                    else:
                        values.extend([value_function(self, cell) for cell in cells])
                    times.extend([t * frame_time] * len(cells))
        return distances, orientations, values, times

    def dipole_fibers_angle_histogram(self):
        """
        Calculate the angle histogram between dipole vectors and fiber orientations.

        Returns:
            angles (list): List of angles between dipole vectors and fiber orientations.
        """
        angles = []
        for split in self.splits:
            dipole = split.dipole_xyz(0)
            if dipole is None:
                continue
            dipole_angle = np.arctan2(dipole[1], dipole[0])
            angles.append(dipole_angle - split.afters.fiber_orientation())

        return angles

    def dipole_main_axis_angle_histogram(self):
        """
        Calculate the angles between the dipole and the main axis for each split.

        Returns:
            angles (list): List of angles between the dipole and the main axis for each split.
        """
        angles = []
        for split in self.splits:
            dipole = split.dipole_xyz(0)
            t = min(split.time_range())
            main_axis = split.dipole_xyz(t)

            angle = np.arctan2(dipole[1], dipole[0]) - np.arctan2(
                main_axis[1], main_axis[0]
            )
            cell_id = split.get_cell_by_time(t)
            before_fiber_orientation = self.dm.cell_attribute(
                cell_id, "fibre_orientation"
            )
            fiber_orientation_diff = (
                split.afters.fiber_orientation() - before_fiber_orientation
            )
            angles.append(angle - fiber_orientation_diff)

        return angles

    def extend_splits(self):
        """
        Extends each split in the list of splits.

        This method iterates over each split in the `splits` list and calls the `extend` method on it.
        The `extend` method is responsible for extending the split.
        """
        for split in self.splits:
            split.extend()

    def find_centers(self):
        """
        Finds the centers of objects in each frame within the specified frame range.

        Returns:
            None
        """
        print("Find centers:")
        for frame_number in tqdm(range(self.frame_range[0], self.frame_range[1] + 1)):
            center = find_center_of_object(self.dm.image(frame_number))
            self.centers.append(center)

    def get_frame_number_by_cell(self, cell_id):
        """
        Get the frame number associated with a given cell ID.

        Parameters:
        - cell_id (int or tuple): The ID(s) of the cell(s) to retrieve the frame number for.

        Returns:
        - frame_number (int): The frame number associated with the given cell ID(s).
        """
        cells = self.dm.cells()
        if isinstance(cell_id, tuple):
            cell_data = cells[cells["cell_id"].isin(cell_id)]
        else:
            cell_data = cells[cells["cell_id"] == cell_id]
        frame_number = cell_data["frame"].values[0]
        return frame_number

    def get_normalize_coord(self, cell_id):
        """
        Get the normalized coordinates of a cell.

        Parameters:
        - cell_id (int): The ID of the cell.

        Returns:
        - x_coord (float): The normalized x-coordinate of the cell.
        - y_coord (float): The normalized y-coordinate of the cell.
        """
        cells = self.dm.cells()
        cell_data = cells[cells["cell_id"] == cell_id]
        frame_number = cell_data["frame"].values[0]
        center = self.centers[frame_number - self.frame_range[0]]
        if center:
            x_coord = cell_data["center_x"].values[0] - center[0]
            y_coord = cell_data["center_y"].values[0] - center[1]
        else:
            x_coord = cell_data["center_x"].values[0]
            y_coord = cell_data["center_y"].values[0]
        return x_coord, y_coord

    def get_orientation(self, cells):
        """
        Calculates the orientation angle between two cells.

        Parameters:
        cells (list): A list of two cells.

        Returns:
        float: The orientation angle in radians.
        """
        x1, y1 = self.get_normalize_coord(cells[0])
        x2, y2 = self.get_normalize_coord(cells[1])
        return np.arctan2(y2 - y1, x2 - x1)

    def get_orientation_from(self, cells_1, cells_2):
        """
        Calculates the orientation angle between two sets of cells.

        Parameters:
        cells_1 (tuple or list): The coordinates of the first set of cells.
        cells_2 (tuple, list, or int): The coordinates of the second set of cells.

        Returns:
        float: The orientation angle in radians.

        """
        x1, y1 = self.get_normalize_coord(cells_1)
        if isinstance(cells_2, list) or isinstance(cells_2, tuple):
            x2, y2 = self.get_normalize_coord_2(cells_2)
        else:
            x2, y2 = self.get_normalize_coord(cells_2)

        return np.arctan2(y2 - y1, x2 - x1)

    def get_normalize_coord_2(self, cell_ids):
        """
        Calculates the normalized coordinates for the average of two given cell IDs.

        Parameters:
            cell_ids (list): A list of two cell IDs.

        Returns:
            numpy.ndarray: The normalized coordinates for the average of the two cell IDs.
        """
        return np.mean(
            [
                self.get_normalize_coord(cell_ids[0]),
                self.get_normalize_coord(cell_ids[1]),
            ],
            axis=0,
        )

    def get_graph_by_cell(self, cell_id):
        """
        Retrieves the graph associated with a specific cell.

        Parameters:
        - cell_id: The ID of the cell.

        Returns:
        - The graph associated with the specified cell.
        """
        frame_number = self.get_frame_number_by_cell(cell_id)
        return self.graphs[frame_number - self.frame_range[0]]

    def find_closest_cell(self, cell_id, frame_number):
        """
        Finds the closest cell to the given cell_id at the specified frame_number.

        Args:
            cell_id (int or tuple): The ID of the cell or a tuple of cell IDs.
            frame_number (int): The frame number to search for the closest cell.

        Returns:
            int: The ID of the closest cell.

        """
        if isinstance(cell_id, tuple):
            return self.find_closest_cell_2(cell_id, frame_number)
        else:
            return self.find_closest_cell_1(cell_id, frame_number)

    def find_closest_cell_1(self, cell_id, frame_number):
        """
        Finds the closest cell to the given cell_id in the specified frame_number.

        Args:
            cell_id (int): The ID of the cell to find the closest cell to.
            frame_number (int): The frame number to search for the closest cell in.

        Returns:
            int or None: The ID of the closest cell, or None if the frame_number is greater than the maximum frame number.

        """
        if frame_number > self.frame_range[1]:
            return None
        min_dist = 30
        closest_cell = None
        graph = self.graph(frame_number)
        for cell in graph.nodes:
            dist = self.how_cells_close(cell_id, cell)
            if dist < min_dist:
                closest_cell = cell
                min_dist = dist
        return closest_cell

    def find_closest_cell_2(self, cell_ids, frame_number):
        """
        Finds the closest cell to the given cell_id in the specified frame_number.

        Args:
            cell_ids (int): The IDs of the cells to find the closest cells to.
            frame_number (int): The frame number to search for the closest cell in.

        Returns:
            int or None: The ID of the closest cell, or None if no cell is found.

        """
        if frame_number > self.frame_range[1]:
            return None
        min_dist = 30
        closest_cell = None
        graph = self.graph(frame_number)
        for cell in graph.edges:
            dist = self.how_cells_close_2(cell_ids, cell)
            if dist < min_dist:
                closest_cell = cell
                min_dist = dist
        return closest_cell

    def how_cells_close(self, cell1, cell2):
        """
        Calculates the closeness between two cells based on various attributes.

        Parameters:
            cell1 (int): The index of the first cell.
            cell2 (int): The index of the second cell.

        Returns:
            float: The closeness between the two cells. Returns np.inf if the cells do not meet the closeness criteria.

        Raises:
            None

        """
        if cell1 < 0 or cell2 < 0:
            return np.inf
        graph_1 = self.get_graph_by_cell(cell1)
        graph_2 = self.get_graph_by_cell(cell2)

        # Check for matching degree:
        if graph_1.degree[cell1] != graph_2.degree[cell2]:
            return np.inf

        # Check for matching brightness, circle and ellipse:
        attributes = ["average_brightness", "circle_identify", "ellipse_identify"]
        for attribute, f in zip(attributes, close_functions):
            attr_1 = graph_1.nodes[cell1][attribute]
            attr_2 = graph_2.nodes[cell2][attribute]
            if f(attr_1, attr_2):
                return np.inf

        # Area difference
        area_1 = self.dm.cell_attribute(cell1, "area")
        area_2 = self.dm.cell_attribute(cell2, "area")
        if np.abs(area_1 - area_2) > Const.max_split_area_diff / 2:
            return np.inf

        x1, y1 = self.get_normalize_coord(cell1)
        x2, y2 = self.get_normalize_coord(cell2)
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    def how_cells_close_2(self, cell1, cell2):
        """
        Calculates the closeness between two cells based on various attributes.

        Parameters:
        - cell1 (tuple): A tuple representing the indices of the first cell.
        - cell2 (tuple): A tuple representing the indices of the second cell.

        Returns:
        - distance (float): The closeness between the two cells.
        """

        if cell1[0] < 0 or cell1[1] < 0 or cell2[0] < 0 or cell2[1] < 0:
            return np.inf
        graph_1 = self.get_graph_by_cell(cell1[0])
        graph_2 = self.get_graph_by_cell(cell2[0])

        # Check for matching degree:
        if np.sum([graph_1.degree[cell] for cell in cell1]) != np.sum(
            [graph_2.degree[cell] for cell in cell2]
        ):
            return np.inf

        # Check for matching brightness, circle and ellipse:
        attributes = ["average_brightness", "circle_identify", "ellipse_identify"]

        for attribute, f in zip(attributes, close_functions):
            attr_1 = (
                graph_1.nodes[cell1[0]][attribute] + graph_1.nodes[cell1[1]][attribute]
            )
            attr_2 = (
                graph_2.nodes[cell2[0]][attribute] + graph_2.nodes[cell2[1]][attribute]
            )
            if f(attr_1, attr_2):
                return np.inf

        # Area difference
        area_1 = np.sum(self.dm.cell_attribute(cell1, "area"))
        area_2 = np.sum(self.dm.cell_attribute(cell2, "area"))
        if np.abs(area_1 - area_2) > Const.max_split_area_diff:
            return np.inf
        orientation_1 = self.get_orientation(cell1)
        orientation_2 = self.get_orientation(cell2)
        if np.abs(orientation_1 - orientation_2) > np.pi / 6:
            return np.inf
        x1, y1 = self.get_normalize_coord_2(cell1)
        x2, y2 = self.get_normalize_coord_2(cell2)
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    def graph(self, frame_number):
        """
        Retrieve the graph corresponding to the given frame number.

        Parameters:
        - frame_number (int): The frame number for which to retrieve the graph.

        Returns:
        - graph: The graph corresponding to the given frame number.
        """
        return self.graphs[frame_number - self.frame_range[0]]

    def create_cells_brightness_image(self, frame_number):
        """
        Calculates the average brightness, ellipse identification, and circle identification
        for each cell in the given frame.

        Args:
            frame_number (int): The frame number for which to calculate the cell properties.

        Returns:
            tuple: A tuple containing three dictionaries:
                - ellipse_identify_per_cell (dict): A dictionary mapping cell IDs to ellipse identification values.
                - circle_identify_per_cell (dict): A dictionary mapping cell IDs to circle identification values.
                - average_brightness_per_cell (dict): A dictionary mapping cell IDs to average brightness values.
        """
        average_brightness_per_cell = dict()
        circle_identify_per_cell = dict()
        ellipse_identify_per_cell = dict()

        image = self.dm.image(frame_number)
        segmentation_image = self.dm.segmentation(frame_number)
        cells = self.dm.cells()
        cells_frame = cells[cells["frame"] == frame_number]
        for cell in cells_frame.to_dict(orient="records"):
            cell_pixels = {(int(cell["center_y"]), int(cell["center_x"]))}
            get_all_pixels_of_cell(
                segmentation_image,
                (int(cell["center_y"]), int(cell["center_x"])),
                cell_pixels,
            )
            y = [pixel[0] for pixel in cell_pixels]
            x = [pixel[1] for pixel in cell_pixels]
            average_brightness_per_cell[cell["cell_id"]] = np.mean(image[y, x])
            if cell["area"] == 0:
                ellipse_identify = 2
                circle_identify = 2
            else:
                if cell["aspect_ratio"] < 2:
                    ellipse_identify = self.ellipse_identify(cell)
                else:
                    ellipse_identify = 2
                circle_identify = cell["perimeter"] / cell["area"] ** 0.5 / (
                    2 * np.pi**0.5
                ) + 2 * (1 / cell["area"] + 2 / cell["perimeter"])

            ellipse_identify_per_cell[cell["cell_id"]] = ellipse_identify
            circle_identify_per_cell[cell["cell_id"]] = circle_identify

        return (
            ellipse_identify_per_cell,
            circle_identify_per_cell,
            average_brightness_per_cell,
        )

    def create_cells_graph(self, frame_number):
        """
        Creates a graph representing the cells and their connections in a specific frame.

        Args:
            frame_number (int): The frame number for which to create the graph.

        Returns:
            networkx.Graph: The graph representing the cells and their connections.
        """

        bond_id_to_cells = dict()
        directed_bonds = self.dm.directed_bonds()
        directed_bonds_frame = directed_bonds[directed_bonds["frame"] == frame_number]
        graph = nx.Graph()

        for record in directed_bonds_frame.to_dict(orient="records"):
            bond_id_to_cells.setdefault(record["bond_id"], []).append(record["cell_id"])

        for _, cells in bond_id_to_cells.items():
            if len(cells) > 1:
                for cell_1, cell_2 in combinations(cells, 2):
                    graph.add_edge(cell_1, cell_2)
            else:
                # edge cell
                graph.add_edge(cells[0], -cells[0])

        (
            ellipse_identify_per_cell,
            circle_identify_per_cell,
            average_brightness_per_cell,
        ) = self.create_cells_brightness_image(frame_number)
        self.add_attributes_to_graph(
            graph,
            ellipse_identify_per_cell,
            circle_identify_per_cell,
            average_brightness_per_cell,
        )

        return graph

    def single_frame_analysis(self, frame_number):
        """
        Perform analysis on a single frame of the cell splits.

        Args:
            frame_number (int): The frame number to analyze.

        Returns:
            tuple: A tuple containing two lists. The first list contains the cells before the split,
                   and the second list contains the cells after the split.
        """
        graph = self.create_cells_graph(frame_number)
        self.graphs.append(graph)
        cells_before_split = self.find_cells_before_split(graph)
        cells_after_split = self.find_cells_after_split(graph)
        return cells_before_split, cells_after_split

    def find_splits(self):
        before_split_log = []
        after_split_log = []

        f_range = range(self.frame_range[0], self.frame_range[1] + 1)

        print(f"Analyzing frames:")
        for frame_number in tqdm(f_range):
            cells_before_split, cells_after_split = self.single_frame_analysis(
                frame_number
            )
            before_split_log.append(cells_before_split)
            after_split_log.append(cells_after_split)

        print(f"Workin on frames:")
        for frame_number in tqdm(f_range):
            index = frame_number - self.frame_range[0]
            self.check_matching_before_and_after(
                before_split_log, after_split_log, index
            )

        self.before_split_log = before_split_log
        self.after_split_log = after_split_log

        self.extend_splits()

    @staticmethod
    def calculate_distance(before, after):
        """
        Calculate the Euclidean distance between the centers of two sets of cells.

        Parameters:
        - before: DataFrame containing the coordinates of the cell centers before splitting.
        - after: DataFrame containing the coordinates of the cell centers after splitting.

        Returns:
        - distance: Euclidean distance between the centers of the two sets of cells.
        """
        x_distance = np.mean(before["center_x"].values) - np.mean(
            after["center_x"].values
        )
        y_distance = np.mean(before["center_y"].values) - np.mean(
            after["center_y"].values
        )

        return np.linalg.norm((x_distance, y_distance))

    @staticmethod
    def calculate_area_diff(before, after):
        return abs(np.sum(before["area"].values) - np.sum(after["area"].values))

    def add_before_after_to_splits(self, before, after):
        """
        Adds the 'before' and 'after' cell IDs to the splits.

        Parameters:
        - before: The cell ID before the split.
        - after: The cell IDs after the split.

        Returns:
        None
        """

        before = before["cell_id"].values[0]
        after = tuple(after["cell_id"].values)
        for split in self.splits:
            if before in split and after in split:
                return
            if before in split:
                split.afters.append(after)
                return
            elif after in split:
                split.befores.append(before)
                return
        split = Split(self)
        split.befores.append(before)
        split.afters.append(after)
        self.splits.append(split)

    def check_matching_before_and_after(self, before_split_log, after_split_log, index):
        """
        Checks for matching cells before and after a split event.

        Args:
            before_split_log (list): List of dictionaries representing the cells before the split event.
            after_split_log (list): List of dictionaries representing the cells after the split event.
            index (int): Index of the split event in the logs.

        Returns:
            None
        """

        before = before_split_log[index]
        after = after_split_log[index]

        before_time_window = before_split_log[max(0, index - Const.time_window) : index]
        if len(after_split_log) - 1 <= index + Const.time_window:
            after_time_window = after_split_log[index + 1 :]
        else:
            after_time_window = after_split_log[index + 1 : index + Const.time_window]
        to_remove = []

        for current_before_index in before:
            for current_after_index in after:
                if current_before_index in current_after_index:
                    to_remove.append(current_before_index)

        for current_before_index in to_remove:
            before.pop(current_before_index, None)

        to_remove = []
        for current_before_index in before:
            remove = True
            for after_time in after_time_window:
                for current_after in after_time.values():
                    if (
                        self.calculate_distance(
                            before[current_before_index],
                            current_after,
                        )
                        < Const.max_split_distace
                        and self.calculate_area_diff(
                            before[current_before_index], current_after
                        )
                        < Const.max_split_area_diff
                    ):
                        self.add_before_after_to_splits(
                            before[current_before_index], current_after
                        )
                        remove = False
                        break
            if remove:

                to_remove.append(current_before_index)

        for current_before_index in to_remove:
            before.pop(current_before_index, None)

        to_remove = []
        for current_after_index in after:
            remove = True

            for time_before in before_time_window:
                for i, current_before in enumerate(time_before.values()):
                    if (
                        self.calculate_distance(
                            current_before,
                            after[current_after_index],
                        )
                        < Const.max_split_distace
                        and self.calculate_area_diff(
                            current_before, after[current_after_index]
                        )
                        < Const.max_split_area_diff
                    ):
                        self.add_before_after_to_splits(
                            current_before, after[current_after_index]
                        )

                        remove = False
                        break
            if remove:
                to_remove.append(current_after_index)

        for current_after_index in to_remove:
            after.pop(current_after_index)

    @staticmethod
    def ellipse_identify(cell):
        """
        Calculates the ellipse identification score for a given cell.

        Parameters:
        - cell (dict): A dictionary containing the properties of the cell.
                       The dictionary should have the following keys:
                       - "perimeter" (float): The perimeter of the cell.
                       - "area" (float): The area of the cell.
                       - "aspect_ratio" (float): The aspect ratio of the cell.

        Returns:
        - float: The ellipse identification score for the cell.
        """

        p = cell["perimeter"]
        s = cell["area"]
        r = cell["aspect_ratio"]

        return (p**2 / s) * (r / (r**2 + 1)) / (2 * np.pi)

    @staticmethod
    def add_attributes_to_graph(
        G,
        ellipse_identify_per_cell,
        circle_identify_per_cell,
        average_brightness_per_cell,
    ):
        """
        Adds attributes to the nodes of a graph.

        Parameters:
        - G: The graph to which attributes will be added.
        - ellipse_identify_per_cell: A dictionary mapping node IDs to ellipse identification values.
        - circle_identify_per_cell: A dictionary mapping node IDs to circle identification values.
        - average_brightness_per_cell: A dictionary mapping node IDs to average brightness values.

        Returns:
        None
        """
        for node in G:
            if node > 0:
                G.nodes[node]["circle_identify"] = circle_identify_per_cell[node]
                G.nodes[node]["average_brightness"] = average_brightness_per_cell[node]
                G.nodes[node]["ellipse_identify"] = ellipse_identify_per_cell[node]
            else:
                G.nodes[node]["ellipse_identify"] = 9
                G.nodes[node]["circle_identify"] = 9
                G.nodes[node]["average_brightness"] = 65535

    def find_cells_before_split(self, graph):
        """
        Finds cells in the graph that meet certain criteria before splitting.

        Args:
            graph (networkx.Graph): The graph representing the cells.

        Returns:
            dict: A dictionary containing the cells that meet the criteria, with the node IDs as keys and cell information as values.
        """
        cells = {}
        for node in graph:
            if (
                graph.nodes[node]["ellipse_identify"] < Const.ellipse_identify_threshold
                and almost_all(
                    graph.nodes[node]["average_brightness"]
                    > Const.brightness_factor_before
                    * graph.nodes[neighbor]["average_brightness"]
                    for neighbor in graph.neighbors(node)
                )
                and all(
                    graph.nodes[node]["average_brightness"]
                    > graph.nodes[neighbor]["average_brightness"]
                    for neighbor in graph.neighbors(node)
                )
                and self.dm.cell_attribute(node, "area") > Const.min_area_before_split
            ):
                cells[node] = self.dm.cells()[self.dm.cells()["cell_id"] == node]
        return cells

    def find_cells_after_split(self, graph):
        """
        Finds cells after split based on the given graph.

        Args:
            graph (networkx.Graph): The graph representing the cells.

        Returns:
            dict: A dictionary containing the cells after split, where the keys are tuples
            representing the node pairs and the values are the corresponding cell data.

        """
        cells = {}
        for node_1, node_2 in graph.edges():
            if node_1 < 0 or node_2 < 0:
                continue
            ci_max = max(
                graph.nodes[node_1]["circle_identify"],
                graph.nodes[node_2]["circle_identify"],
            )
            ci_min = min(
                graph.nodes[node_1]["circle_identify"],
                graph.nodes[node_2]["circle_identify"],
            )
            average_brightness_min = min(
                graph.nodes[node_1]["average_brightness"],
                graph.nodes[node_2]["average_brightness"],
            )
            neighnors = (
                set(graph.neighbors(node_1)) | set(graph.neighbors(node_2))
            ) - ({node_1} | {node_2})
            if (
                all(
                    average_brightness_min
                    > Const.brightness_factor_after
                    * graph.nodes[neighbor]["average_brightness"]
                    for neighbor in neighnors
                )
                and self.calculate_area_diff(
                    self.dm.cells()[self.dm.cells()["cell_id"] == node_1],
                    self.dm.cells()[self.dm.cells()["cell_id"] == node_2],
                )
                < Const.max_after_split_area_diff
                and ci_max < Const.circle_identify_threshold
            ):

                cells[(node_1, node_2)] = self.dm.cells()[
                    (self.dm.cells()["cell_id"] == node_1)
                    | (self.dm.cells()["cell_id"] == node_2)
                ]
        return cells
