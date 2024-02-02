import numpy as np
from typing import Tuple, Any
import pandas as pd
import streamlit as st
import pedpy
import pandas as pd
from shapely.geometry import Polygon
from scipy.spatial import KDTree
from typing import Tuple

import plots as pl


def generate_oval_shape_points(
    num_points: int,
    length: float = 2.3,
    radius: float = 1.65,
    start: Tuple[float, float] = (0.0, 0.0),
    dx: float = 0.2,
    threshold: float = 0.5,
):
    """Generate points on a closed setup with two segments and two half circles."""
    points = [start]
    selected_points = [start]
    last_selected = start  # keep track of the last selected point

    # Define the points' generating functions
    def dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    # Calculate dphi based on the dx and radius
    dphi = dx / radius

    center2 = (start[0] + length, start[1] + radius)
    center1 = (start[0], start[1] + radius)

    npoint_on_segment = int(length / dx)

    # first segment
    for i in range(1, npoint_on_segment + 1):
        tmp_point = (start[0] + i * dx, start[1])
        points.append(tmp_point)
        if dist(tmp_point, last_selected) >= threshold:
            selected_points.append(tmp_point)
            last_selected = tmp_point

    # first half circle
    for phi in np.arange(-np.pi / 2, np.pi / 2, dphi):
        x = center2[0] + radius * np.cos(phi)
        y = center2[1] + radius * np.sin(phi)
        tmp_point = (x, y)
        points.append(tmp_point)
        if dist(tmp_point, last_selected) >= threshold:
            selected_points.append(tmp_point)
            last_selected = tmp_point

    # second segment
    for i in range(1, npoint_on_segment + 1):
        tmp_point = (
            start[0] + (npoint_on_segment + 1) * dx - i * dx,
            start[1] + 2 * radius,
        )
        points.append(tmp_point)
        if dist(tmp_point, last_selected) >= threshold:
            selected_points.append(tmp_point)
            last_selected = tmp_point

    # second half circle
    for phi in np.arange(np.pi / 2, 3 * np.pi / 2 - dphi, dphi):
        x = center1[0] + radius * np.cos(phi)
        y = center1[1] + radius * np.sin(phi)
        tmp_point = (x, y)
        points.append(tmp_point)
        if dist(tmp_point, last_selected) >= threshold:
            selected_points.append(tmp_point)
            last_selected = tmp_point

    if dist(selected_points[-1], start) < threshold:
        selected_points.pop()
    if num_points > len(selected_points):
        print(f"warning: {num_points} > Allowed: {len(selected_points)}")

    selected_points = selected_points[:num_points]
    return points, selected_points


def generate_parcour():
    _, exterior = generate_oval_shape_points(
        num_points=50,
        radius=1.65 + 0.4,
        # start=(0.0, -2 + 1.6),
        start=(-1, -2),
        threshold=0.2,
    )
    _, interior = generate_oval_shape_points(
        num_points=50,
        radius=1.65 - 0.4,
        length=2,
        # start=(0, -1.2 + 1.6),
        start=(-1, -1.2),
        threshold=0.2,
    )

    return exterior, interior


def sorting_key(filename):
    if filename.startswith("female"):
        return (0, filename)
    elif filename.startswith("male"):
        return (1, filename)
    elif filename.startswith("mix_sorted"):
        return (2, filename)
    elif filename.startswith("mix_random"):
        return (3, filename)
    else:
        return (4, filename)  # For filenames that don't match any category


def rename_columns(data: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns of the dataframe based on the given mapping.
    """
    return data.rename(columns=mapping, inplace=True)


def set_column_types(data: pd.DataFrame, col_types: dict[str, Any]) -> pd.DataFrame:
    """
    Set the types of the dataframe columns based on the given column types.
    """
    # Ensure columns are in data before type casting
    valid_types = {
        col: dtype for col, dtype in col_types.items() if col in data.columns
    }
    return data.astype(valid_types)


def calculate_fps(data: pd.DataFrame) -> int:
    """
    Calculate fps based on the mean difference of the 'time' column.
    """
    mean_diff = data.groupby("id")["time"].diff().dropna().mean()
    return int(round(1 / mean_diff))


def load_file(file):
    data = pd.read_csv(file)
    rename_columns(data, st.session_state.config.rename_mapping)
    set_column_types(data, st.session_state.config.column_types)
    fps = calculate_fps(data)
    trajectory_data = pedpy.TrajectoryData(data=data, frame_rate=fps)
    return trajectory_data


def rotate_trajectories(df, shift_x, shift_y, angle_degrees):
    """
    Rotates the x and y coordinates in the dataframe around a center point by a specified angle.

    Parameters:
    df (pd.DataFrame): Dataframe containing the trajectories with columns 'x' and 'y'.
    center_x (float): x-coordinate of the center of rotation.
    center_y (float): y-coordinate of the center of rotation.
    angle_degrees (float): Angle in degrees by which to rotate the trajectories.

    Returns:
    pd.DataFrame: A new dataframe with rotated coordinates.
    """
    angle_radians = np.radians(angle_degrees)

    center_x = 0
    center_y = 0
    x_shifted = df["x"] - center_x
    y_shifted = df["y"] - center_y

    df_rotated = df.copy()

    df_rotated["x"] = (
        shift_x
        + center_x
        + x_shifted * np.cos(angle_radians)
        - y_shifted * np.sin(angle_radians)
    )
    df_rotated["y"] = (
        shift_y
        + center_y
        + x_shifted * np.sin(angle_radians)
        + y_shifted * np.cos(angle_radians)
    )

    return df_rotated


def get_neighbors_at_frame(
    frame: int, df: pd.DataFrame, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the distances and indices of the k nearest neighbors for each point at a given frame.

    Parameters:
    frame (int): The frame number to filter data.
    df (DataFrame): The DataFrame containing the data.
                    It should have columns for 'frame', 'x', and 'y' coordinates.
    k (int): The number of nearest neighbors to find.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays. The first array
                                   contains the distances to the k nearest neighbors for each point,
                                   and the second array contains the indices of these neighbors.
    """
    # Filter the DataFrame for the specified frame
    at_frame = df[df["frame"] == frame]

    # Extract points as a 2D NumPy array
    points = at_frame[["x", "y"]].to_numpy()

    # Use KDTree for finding nearest neighbors
    tree = KDTree(points)
    if k < len(points):
        nearest_dist, nearest_ind = tree.query(points, k)
        return nearest_dist, nearest_ind

    return np.array([]), np.array([])


def get_neighbors_special_agent_data(
    agent: int,
    frame: int,
    df: pd.DataFrame,
    nearest_dist: np.ndarray,
    nearest_ind: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    # Filter DataFrame for the specified fram
    frames = df["frame"].to_numpy()
    first_frame = frames[0]
    at_frame = df[(df["frame"] == frame)]

    # Extract points, speeds, and ids
    points = at_frame[["x", "y"]].to_numpy()
    point_agent = df[(df["frame"] == frame) & (df["id"] == agent)][
        ["x", "y"]
    ].to_numpy()
    point_agent_future = df[(df["frame"] == frame + 50) & (df["id"] == agent)][
        ["x", "y"]
    ].to_numpy()

    Ids = at_frame["id"].to_numpy()
    neighbor_type = ["next", "prev"]
    # Check if the agent is in the current frame
    if agent in Ids:
        agent_index = np.where(Ids == agent)[0][0]
        mask = nearest_ind[:, 0] == agent_index
        neighbors_ind = nearest_ind[mask][0, 1:]
        neighbors_dist = nearest_dist[mask][0, 1:]
        neighbors = np.array([points[i] for i in neighbors_ind])
        neighbors_ids = np.array([Ids[i] for i in neighbors_ind])
    else:
        return np.array([]), np.array([]), 0, np.array([]), np.array([])

    if frame == first_frame:
        distance_now = (neighbors[0][0] - point_agent[0][0]) ** 2 + (
            neighbors[0][1] - point_agent[0][1]
        ) ** 2
        distance_future = (neighbors[0][0] - point_agent_future[0][0]) ** 2 + (
            neighbors[0][1] - point_agent_future[0][1]
        ) ** 2

        if distance_future > distance_now:
            # neighbors = neighbors[::-1]
            # neighbors_ids = neighbors_ids[::-1]
            neighbor_type = ["prev", "next"]

    # Calculate area if there are enough neighbors
    if len(neighbors) > 2:
        my_polygon = Polygon(neighbors)
        neighbors = np.array(my_polygon.exterior.coords)
        area = my_polygon.area
    else:
        area = 0

    return neighbors, neighbors_ids, area, neighbors_dist, neighbor_type


def plot_neighbors_analysis(data, ids, exterior, interior):
    n0, n1, n2 = st.columns((1, 1, 1))
    agent = n1.number_input(
        "Agent",
        min_value=int(min(ids)),
        max_value=int(max(ids)),
        value=int(min(ids)),
        placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
        format="%d",
    )
    frames = data["frame"].to_numpy()

    frame = n2.number_input(
        "Frame",
        int(frames[0]),
        int(frames[-1]),
        int(frames[0]),
        step=1,
        help="Frame at which to display a snapshot of the neighbors",
    )

    k = n0.number_input(
        "k neighbors",
        2,
        4,
        3,
        format="%d",
    )
    rotated_data = rotate_trajectories(
        data,
        st.session_state.center_x,
        st.session_state.center_y,
        st.session_state.angle_degrees,
    )
    nearest_dist, nearest_ind = get_neighbors_at_frame(frame, rotated_data, k)
    (
        neighbors,
        neighbors_ids,
        area,
        agent_distances,
        neighbor_type,
    ) = get_neighbors_special_agent_data(
        agent, frame, rotated_data, nearest_dist, nearest_ind
    )
    fig = pl.plot_agent_and_neighbors(
        agent,
        frame,
        rotated_data,
        neighbors,
        neighbors_ids,
        exterior,
        interior,
        neighbor_type,
    )

    return fig


def get_numbers_country(country):
    n_female = 0
    n_male = 0
    n_mixed_random = 0
    n_mixed_sorted = 0
    files = st.session_state.config.files[country]
    for file in files:
        name = file.split("/")[-1]
        if name.startswith("female"):
            n_female += 1
        if name.startswith("male"):
            n_male += 1
        if name.startswith("mix_sorted"):
            n_mixed_sorted += 1
        if name.startswith("mix_random"):
            n_mixed_random += 1

    return n_female, n_male, n_mixed_random, n_mixed_sorted
