import numpy as np
import networkx as nx

from utils import (
    color_cells,
    get_all_pixels_of_cell,
    get_main_axis,
    add_fiber_orientation_line,
    add_main_axis_line,
)


class GeneralSplit(list):
    """
    A class representing a general split of cells.

    This class extends the built-in `list` class and provides additional methods for cell splitting operations.
    """

    def __init__(self, cs):
        super().__init__()
        self.cs = cs
        self.max_skip = 1

    def dipole_xyz(self, cell_ids):
        raise NotImplementedError

    def center_xyz(self, t):
        raise NotImplementedError

    def frames(self):
        return [self.cs.get_frame_number_by_cell(cell) for cell in self]

    @staticmethod
    def next_frame(frame_number):
        raise NotImplementedError

    def time_ok(self, t):
        raise NotImplementedError

    def t_range(self):
        raise NotImplementedError

    def fiber_orientation(self, t=None):
        """
        Calculate the average fiber orientation of the cells at a given time point.

        Parameters:
            t (float): The time point at which to calculate the fiber orientation. If None, the average fiber orientation
                       across all time points will be calculated.

        Returns:
            float: The average fiber orientation at the specified time point, or across all time points if t is None.
            None: If the time point is invalid or the cell IDs are not tuples.

        """
        if t is None:
            return np.mean([self.fiber_orientation(t) for t in self.t_range()])
        if not self.time_ok(t):
            return None
        cell_ids = self[t]
        if not isinstance(cell_ids, tuple):
            cell_ids = (cell_ids,)
        return np.mean(self.cs.dm.cell_attribute(cell_ids, "fibre_orientation"))

    def extend_in_time(self):
        """
        Extends the cells in time by finding the closest cell in the previous frames.

        This method iterates over the cells in the current frame and searches for the closest cell in the previous/next frames.
        If a closest cell is found, it is added to the list of new cells. The process continues until no closest cell is found
        or the maximum skip count is reached.

        Returns:
            None

        Raises:
            None
        """
        new_cells = []
        self.sort(key=self.cs.get_frame_number_by_cell)
        frames_numbers = self.frames()
        for cell, frame_number in zip(self, frames_numbers):
            skip_count = 0
            while True:
                if frame_number - 1 in frames_numbers:
                    break
                closest_cell = self.cs.find_closest_cell(
                    cell, self.next_frame(frame_number)
                )
                if closest_cell is None:
                    if skip_count < self.max_skip:
                        skip_count += 1
                        frame_number = self.next_frame(frame_number)
                        continue
                    break
                new_cells.append(closest_cell)
                cell = closest_cell
                frame_number -= 1
        self.extend(new_cells)
        self.sort(key=self.cs.get_frame_number_by_cell)


class Befores(GeneralSplit):
    """
    Represents a class for handling 'before' frames in cell splitting.

    Inherits from the GeneralSplit class.

    """

    def __init__(self, cs):
        super().__init__(cs)

    def time_ok(self, t):
        return t < 0 and -len(self) <= t

    @staticmethod
    def next_frame(frame_number):
        return frame_number - 1

    def t_range(self):
        return range(-len(self), 0)

    def center_xyz(self, t):
        if len(self) >= t:
            return None
        cell_id = self[t]
        cells = self.cs.dm.cells()
        cell_data = cells[cells["cell_id"] == cell_id].iloc[0]
        return np.array(
            [cell_data["center_x"], cell_data["center_y"], cell_data["center_z"]]
        )

    def dipole_xyz(self, cell_id):
        cell = self.cs.dm.cells()[self.cs.dm.cells()["cell_id"] == cell_id].iloc[0]
        frame_number = self.cs.get_frame_number_by_cell(cell_id)
        segmentation_image = self.cs.dm.segmentation(frame_number)

        cell_pixels = {(int(cell["center_y"]), int(cell["center_x"]))}
        get_all_pixels_of_cell(
            segmentation_image,
            (int(cell["center_y"]), int(cell["center_x"])),
            cell_pixels,
        )
        return get_main_axis(list(cell_pixels))


class Afters(GeneralSplit):
    """
    A class representing the Afters split.

    Inherits from the GeneralSplit class.
    """

    def __init__(self, cs):
        super().__init__(cs)

    def time_ok(self, t):
        return t >= 0 and t < len(self)

    @staticmethod
    def next_frame(frame_number):
        return frame_number + 1

    def t_range(self):
        return range(len(self))

    def dipole_xyz(self, cell_ids):
        """
        Calculates the dipole vector for the given cell IDs.

        Args:
            cell_ids (list): A list of cell IDs.

        Returns:
            numpy.ndarray: The normalized dipole vector.
        """
        cells = self.cs.dm.cells()
        cell_data = cells[cells["cell_id"].isin(cell_ids)]
        centers = cell_data[["center_x", "center_y", "center_z"]].values
        dipole = np.array(centers[1] - centers[0])
        normed_dipole = dipole / np.linalg.norm(dipole) * np.sign(dipole[1])
        return normed_dipole

    def center_xyz(self, t):
        """
        Calculates the center coordinates for the given time value.

        Args:
            t (int): The time value.

        Returns:
            numpy.ndarray: The mean center coordinates.
        """
        if not self.time_ok(t):
            return None
        cell_ids = self[t]
        cells = self.cs.dm.cells()
        cell_data = cells[cells["cell_id"].isin(cell_ids)]
        return cell_data[["center_x", "center_y", "center_z"]].mean().values


class Split:
    """
    Represents a split event in a cell lineage.

    Attributes:
        max_skip (int): The maximum number of frames to skip when extending the split.
        cs (CellSplit): The CellSplit object associated with the split.
        befores (Befores): The Befores object containing the cells before the split.
        afters (Afters): The Afters object containing the cells after the split.
    """

    def __init__(self, cs):
        self.max_skip = 1
        self.cs = cs
        self.befores = Befores(cs)
        self.afters = Afters(cs)

    def __contains__(self, other):
        """
        Check if the split contains a given cell.

        Args:
            other: The cell to check.

        Returns:
            bool: True if the cell is in the split, False otherwise.
        """
        if isinstance(other, tuple):
            return other in self.afters
        return other in self.befores

    def extend(self):
        """
        Extend the split by adding cells in time.
        """
        self.remove_before_that_after_after()
        self.befores.extend_in_time()
        self.afters.extend_in_time()

    def remove_before_that_after_after(self):
        """
        Remove cells from befores that appear after the first appearance of cells in afters.
        """
        to_remove = []
        for before in self.befores:
            if self.cs.get_frame_number_by_cell(before) > self.split_first_appereance():
                to_remove.append(before)

        for before in to_remove:
            self.befores.remove(before)

    def split_first_appereance(self):
        """
        Get the frame number of the first appearance of cells in afters.

        Returns:
            int: The frame number.
        """
        return self.cs.get_frame_number_by_cell(self.afters[0])

    def get_cells_with_distance_r(self, split_t, r):
        """
        Get cells with a distance of r from the split at a given time.

        Args:
            split_t (int): The time offset from the split.
            r (int): The distance from the split.

        Returns:
            list: The cells with the specified distance from the split.
        """
        origin_cells = self.get_cell_by_time(split_t)
        if origin_cells is None:
            return None
        elif not isinstance(origin_cells, tuple):
            origin_cells = (origin_cells,)
        frame_number = self.cs.get_frame_number_by_cell(origin_cells[0])
        graph = self.cs.graph(frame_number)

        return [
            node
            for node in graph.nodes()
            if nx.has_path(graph, node, origin_cells[0])
            and min(
                nx.shortest_path_length(graph, node, origin_cell)
                for origin_cell in origin_cells
            )
            == r
            and node > 0
        ]

    def frames(self):
        """
        Get the frames associated with the split.

        Returns:
            list: The frames.
        """
        return self.befores.frames() + self.afters.frames()

    def time_range(self):
        """
        Get the time range of the split.

        Returns:
            range: The time range.
        """
        frames = self.frames()
        split_frame = self.split_first_appereance()
        return range(frames[0] - split_frame, frames[-1] - split_frame + 1)

    def time_ok(self, t):
        """
        Check if a given time offset is valid for the split.

        Args:
            t (int): The time offset.

        Returns:
            bool: True if the time offset is valid, False otherwise.
        """
        frame_number = self.split_first_appereance() + t
        return frame_number in self.frames()

    def dipole_xyz(self, t):
        """
        Get the dipole coordinates of cells at a given time.

        Args:
            t (int): The time offset.

        Returns:
            tuple: The dipole coordinates (x, y, z) or None if the cells are not found.
        """
        cell_ids = self.get_cell_by_time(t)
        if cell_ids is None:
            return None

        if not isinstance(cell_ids, tuple) and cell_ids in self.befores:
            return self.befores.dipole_xyz(cell_ids)
        elif cell_ids in self.afters:
            return self.afters.dipole_xyz(cell_ids)
        return None

    def _get_cell_by_frame(self, frame_number):
        """
        Get the cell at a given frame number.

        Args:
            frame_number (int): The frame number.

        Returns:
            tuple: The cell or None if not found.
        """
        if frame_number in self.befores.frames():
            return self.befores[self.befores.frames().index(frame_number)]
        if frame_number in self.afters.frames():
            return self.afters[self.afters.frames().index(frame_number)]
        return None

    def get_cell_by_time(self, t):
        """
        Get the cell at a given time offset.

        Args:
            t (int): The time offset.

        Returns:
            tuple: The cell or None if not found.
        """
        frame_number = self.split_first_appereance() + t
        return self._get_cell_by_frame(frame_number)

    def create_split_video(
        self,
        fps=2,
        use_color=True,
        plot_fiber_orientation=True,
        plot_main_axis=True,
    ):
        """
        Create a video of the split event.

        Args:
            fps (int): The frames per second of the video.
            use_color (bool): Whether to use color for cell visualization.
            plot_fiber_orientation (bool): Whether to plot fiber orientation lines.
            plot_main_axis (bool): Whether to plot main axis lines.
        """
        cells = self.cs.dm.cells()
        first_frame = self.befores.frames()[0]
        output_folder = self.cs.output_folder / "splits"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_folder
            / f"{first_frame}_{self.befores[0]}_{len(self.befores) + len(self.afters)}.avi"
        )
        video_writer = self.cs.dm.creat_video_writer(output_path, fps)

        for cells_ids, frame_number in zip(self.befores + self.afters, self.frames()):
            if not isinstance(cells_ids, tuple) == 1:
                cells_ids = (cells_ids,)
                cell_color = (0, 0, 255)
            else:
                cell_color = (255, 0, 0)
            image = self.cs.dm.image_for_video(frame_number)
            cell_data = cells[cells["cell_id"].isin(cells_ids)]
            if use_color:
                segmentation = self.cs.dm.segmentation(frame_number)
                image = color_cells(image, segmentation, cell_data, cell_color)
            if plot_fiber_orientation:
                add_fiber_orientation_line(image, cell_data, color=(255, 255, 255))
            if plot_main_axis:
                add_main_axis_line(self.cs, image, cell_data, color=(0, 255, 0))
            self.cs.dm.save_image_for_video(image, frame_number)
            video_writer.write(image)

        video_writer.release()
