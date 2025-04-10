import numpy as np
import matplotlib.pyplot as plt
import cv2

MAX = 800


def almost_all(iterable):
    """
    Checks if almost all elements in the iterable are truthy.

    Args:
        iterable: An iterable object containing elements to be checked.

    Returns:
        bool: True if almost all elements are truthy, False otherwise.
    """
    return sum(not bool(x) for x in iterable) <= 1


def get_all_pixels_of_cell(segmentation_image, pixel, visited_pixels, count=0):
    """
    Recursively finds all the pixels of a cell in a segmentation image.

    Args:
        segmentation_image (numpy.ndarray): The segmentation image.
        pixel (tuple): The starting pixel coordinates.
        visited_pixels (set): A set of visited pixels.
        count (int, optional): The current count of pixels visited. Defaults to 0.

    Returns:
        None, but the visited_pixels set is updated with the pixels of the cell.

    """
    if count > MAX:
        return
    next_pixels = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for direction in directions:
        new_pixel = (pixel[0] + direction[0], pixel[1] + direction[1])
        if (
            new_pixel[0] >= segmentation_image.shape[0]
            or new_pixel[1] >= segmentation_image.shape[1]
            or new_pixel[0] < 0
            or new_pixel[1] < 0
        ):
            continue

        if (
            segmentation_image[new_pixel[0], new_pixel[1]] == 0
            and new_pixel not in visited_pixels
        ):
            visited_pixels.add(new_pixel)
            next_pixels.append(new_pixel)
    for next_pixel in next_pixels:
        get_all_pixels_of_cell(
            segmentation_image, next_pixel, visited_pixels, count + 1
        )


def get_mask(segmentation_image, cell):
    """
    Generates a binary mask for a given cell in a segmentation image.

    Args:
        segmentation_image (numpy.ndarray): The segmentation image.
        cell (dict): A dictionary containing information about the cell, including its center coordinates.

    Returns:
        numpy.ndarray: A binary mask where the pixels corresponding to the cell are set to 1.

    """
    loc = (
        int(cell["center_y"]),
        int(cell["center_x"]),
    )
    cell_pixels = {loc}
    get_all_pixels_of_cell(
        segmentation_image,
        loc,
        cell_pixels,
    )
    mask = np.zeros(segmentation_image.shape)
    for pixel in cell_pixels:
        mask[pixel] = 1
    return mask


def apply_mask(image, mask, color, alpha=0.2):
    """
    Applies a mask to an image by blending it with a specified color.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask to be applied.
        color (tuple): The color to blend with the image.
        alpha (float, optional): The blending factor. Defaults to 0.2.

    Returns:
        numpy.ndarray: The resulting image with the mask applied.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c],
        )

    return image


def color_cells(image, segmentation, cells, color):
    """
    Color the cells in the given image based on the provided segmentation and cell information.

    Args:
        image (numpy.ndarray): The input image.
        segmentation (numpy.ndarray): The segmentation mask.
        cells (pandas.DataFrame): The cell information.
        color (tuple): The color to apply to the cells.

    Returns:
        numpy.ndarray: The image with colored cells.
    """
    for cell in cells.to_dict("records"):
        mask = get_mask(segmentation, cell)
        image = apply_mask(image, mask, color)

    return image


def add_fiber_orientation_line(image, cells, color=(255, 255, 255), thickness=2):
    """
    Adds a line representing the fiber orientation to the given cells.

    Args:
        image (numpy.ndarray): The input image.
        cells (pandas.DataFrame): DataFrame containing the cells information.
        color (tuple, optional): The color of the line. Defaults to (255, 255, 255).
        thickness (int, optional): The thickness of the line. Defaults to 2.

    Returns:
        numpy.ndarray: The image with the fiber orientation line added.
    """
    fiber_orientation = cells["fibre_orientation"].mean()
    center_x = cells["center_x"].mean()
    center_y = cells["center_y"].mean()
    length = 25
    start = (
        int(center_x - length * np.cos(fiber_orientation)),
        int(center_y - length * np.sin(fiber_orientation)),
    )
    end = (
        int(center_x + length * np.cos(fiber_orientation)),
        int(center_y + length * np.sin(fiber_orientation)),
    )

    cv2.line(image, start, end, color, thickness)
    return image


def add_main_axis_line(cs, image, cells, color=(255, 255, 255), thickness=2):
    """
    Adds a main axis line to the given image based on the provided cells.

    Parameters:
    - cs: An object representing the cell splitting.
    - image: The image to which the main axis line will be added.
    - cells: A DataFrame containing information about the cells.
    - color: The color of the main axis line (default: white).
    - thickness: The thickness of the main axis line (default: 2).

    Returns:
    - The image with the main axis line added.
    """
    if len(cells) == 1:
        cell = cells.iloc[0]
        cell_id = cells["cell_id"].values[0]
        frame_number = cs.get_frame_number_by_cell(cell_id)
        segmentation_image = cs.dm.segmentation(frame_number)

        cell_pixels = {(int(cell["center_y"]), int(cell["center_x"]))}
        get_all_pixels_of_cell(
            segmentation_image,
            (int(cell["center_y"]), int(cell["center_x"])),
            cell_pixels,
        )
        x, y = get_main_axis(list(cell_pixels))
    else:
        cell_1 = cells.iloc[0]
        cell_2 = cells.iloc[1]
        x = cell_2["center_x"] - cell_1["center_x"]
        y = cell_2["center_y"] - cell_1["center_y"]

    angle = np.arctan2(y, x)

    center_x = cells["center_x"].mean()
    center_y = cells["center_y"].mean()
    length = 25
    start = (
        int(center_x - length * np.cos(angle)),
        int(center_y - length * np.sin(angle)),
    )
    end = (
        int(center_x + length * np.cos(angle)),
        int(center_y + length * np.sin(angle)),
    )

    cv2.line(image, start, end, color, thickness)
    return image


def find_center_of_object(gray):
    """
    Finds the center coordinates of the largest object in a grayscale image.

    Parameters:
    gray (numpy.ndarray): The grayscale image.

    Returns:
    tuple: A tuple containing the x and y coordinates of the center of the object.
           If no object is found, None is returned.
    """
    gray = (gray / 256).astype("uint8")

    # Apply binary threshold
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assuming the largest contour is the object
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments
        M = cv2.moments(largest_contour)

        # Calculate center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        return cX, cY
    else:
        return None


def get_main_axis(object_coords):
    """
    Calculates the main axis of an object based on its coordinates.

    Parameters:
    object_coords (list): List of coordinates of the object.

    Returns:
    numpy.ndarray: The main axis vector of the object.

    """
    object_coords = np.array(object_coords).reshape(-1, 1, 2)
    fitted_ellipse = cv2.fitEllipse(object_coords)

    fitted_angle = np.deg2rad(fitted_ellipse[2]) + np.pi / 2
    main_axis = np.array([np.sin(fitted_angle), np.cos(fitted_angle)])

    return main_axis * np.sign(main_axis[1])


def get_Q_tensor(cs, cell_id):
    """
    Calculate the Q tensor for a given cell.

    Parameters:
    cs (CellSplit): The CellSplit object containing the cell data.
    cell_id (int): The ID of the cell.

    Returns:
    Q (numpy.ndarray): The Q tensor as a 2x2 numpy array.
    None: If the q_xx or q_xy attributes are missing for the cell.
    """
    q_xx = cs.dm.cell_attribute(cell_id, "q_xx")
    q_xy = cs.dm.cell_attribute(cell_id, "q_xy")
    if q_xx is None or q_xy is None:
        return None
    Q = np.array([[q_xx, q_xy], [q_xy, -q_xx]])
    return Q


def project_shape_tensor(Q, axis):
    """
    Projects a shape tensor onto a given axis.

    Parameters:
    - Q (numpy.ndarray): The shape tensor to be projected.
    - axis (list or numpy.ndarray): The axis onto which the shape tensor is projected.

    Returns:
    - projection (float): The projection of the shape tensor onto the given axis.
    """

    # Normalize the axis vector
    axis = np.array(axis)

    # Calculate the projection
    projection = axis.T @ Q @ axis

    return projection


def create_colored_ploar(r, theta, z, time, folder_path):
    """
    Create colored polar plots based on the given data.

    Args:
        r (list or numpy.ndarray): List or array of radial distances.
        theta (list or numpy.ndarray): List or array of angular coordinates.
        z (list or numpy.ndarray): List or array of z-coordinates.
        time (list or numpy.ndarray): List or array of time values.
        folder_path (str or pathlib.Path): Path to the folder where the plots will be saved.

    Returns:
        None
    """
    folder_path.mkdir(parents=True, exist_ok=True)

    r = np.array(r)
    theta = np.array(theta)
    z = np.array(z)
    time = np.array(time)

    # Remove NaN values
    mask = np.isnan(z)
    r = r[~mask]
    theta = theta[~mask]
    z = z[~mask]
    time = time[~mask]

    num_theta_bins = 6
    theta_bins = np.linspace(-np.pi / 2, np.pi / 2, num_theta_bins + 1)

    max_r = np.max(r)
    num_r_bins = max_r + 1
    r_bins = np.linspace(0, max_r + 1, num_r_bins + 1)

    theta = np.mod(theta + np.pi / 2, np.pi) - np.pi / 2

    theta_types = ["regular", "fold"]
    for theta_type in theta_types:
        vmin = None
        vmax = None
        folder = folder_path / theta_type
        folder.mkdir(parents=True, exist_ok=True)
        for index, t in enumerate([0] + list(sorted(np.unique(time)))):
            mask_t = time == t
            r_t = r[mask_t]
            theta_t = theta[mask_t]
            z_t = z[mask_t]

            # Digitize the data points into the bins
            theta_digitized = np.digitize(theta_t, theta_bins) - 1
            r_digitized = np.digitize(r_t, r_bins) - 1

            # Create an array to store the mean values for each bin
            mean_values = np.zeros((num_r_bins, num_theta_bins))

            # Calculate the mean value for each bin
            for i in range(num_r_bins):
                for j in range(num_theta_bins):
                    mask = (r_digitized == i) & (theta_digitized == j) | (
                        (r_digitized == 0) & (i == 0)
                    )
                    if np.any(mask):
                        mean_values[i, j] = np.mean(z_t[mask])
                    else:
                        mean_values[i, j] = np.nan  # If no points in bin, set as NaN
            if vmin is None and vmax is None:
                vmin = np.min(mean_values)
                vmax = np.max(mean_values)
                continue
            # Create the polar plot
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
            ax.set_thetagrids(
                [0, 30, 60, 90, 270, 300, 330],
                labels=["0", "30", "60", "90", "-90", "-60", "-30"],
            )
            plt.subplots_adjust(right=1.5)

            # Create the color mesh
            r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
            theta_bin_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
            Theta, R = np.meshgrid(theta_bin_centers, r_bin_centers)

            c = ax.pcolormesh(
                Theta, R, mean_values, vmin=vmin, vmax=vmax, shading="auto", cmap="jet"
            )

            # Add a color bar
            plt.colorbar(c)

            # Show the plot
            splits_count = int(sum(r_t == 0) / (int(t >= 0) + 1))
            file_path_t = folder / f"i_{index:02}_t_{t}_{splits_count}.png"
            plt.savefig(file_path_t, bbox_inches="tight")

            # Close the plot to free up memory
            plt.close()
        theta = np.concatenate([np.abs(theta), -np.abs(theta)])
        z = np.concatenate([z, z])
        r = np.concatenate([r, r])
        time = np.concatenate([time, time])


def create_polar_histogram(theta, file_path=None):

    theta = np.array(theta)
    theta = np.mod(theta + np.pi / 2, np.pi) - np.pi / 2
    # Create the histogram
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
    ax.set_thetagrids(
        [0, 30, 60, 90, 270, 300, 330],
        labels=["0", "30", "60", "90", "-90", "-60", "-30"],
    )

    num_bins = 9
    counts, bin_edges = np.histogram(
        theta, bins=num_bins, range=(-np.pi / 2, np.pi / 2)
    )
    bin_width = (np.pi) / num_bins
    heights = np.sqrt(counts)
    bin_centers = bin_edges[:-1] + bin_width / 2
    ax.bar(
        bin_centers,
        heights,
        width=bin_width,
        bottom=0,
        color="skyblue",
        edgecolor="black",
    )
    # Create the histogram
    # n, bins, patches = ax.hist(theta, bins="auto", color="skyblue", edgecolor="black")

    # Show the plot
    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    else:
        plt.show()

    # Close the plot to free up memory
    plt.close()
