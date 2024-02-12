"""Collection of some helpful functions."""

from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import pedpy
import requests
import streamlit as st
from plotly.graph_objs import Figure
from scipy.spatial import KDTree
from shapely.geometry import Polygon

import plots as pl


def set_rotation_variables(selected_file, country):
    ger_substrings_to_check = [
        "female",
        "20_19",
        "16_18",
        "40_11",
        "8_16",
        "8_17",
        "32_11",
        "36_11",
        "4_11",
        "4_12",
        "4_13",
        "4_15",
        "4_14",
    ]
    aus_substrings_to_check = ["female", "mix_sorted_40_01", "mix_random_41_01"]
    if country == "ger":
        if any(substring in selected_file for substring in ger_substrings_to_check):
            st.session_state.center_x = 3.1
            st.session_state.center_y = 3
            st.session_state.angle_degrees = 90
        else:
            st.session_state.center_x = 3.1
            st.session_state.center_y = -3
            st.session_state.angle_degrees = 90

    if country == "aus":
        if any(substring in selected_file for substring in aus_substrings_to_check):
            st.session_state.center_x = 1.7
            st.session_state.center_y = -0.2
            st.session_state.angle_degrees = 85
        else:
            st.session_state.center_x = 1.8
            st.session_state.center_y = -6.3
            st.session_state.angle_degrees = 89

    if country == "chn":
        if "female" in selected_file:
            st.session_state.center_x = 0.1
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 87
        elif "mix_random" in selected_file:
            st.session_state.center_x = 0.1
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 85
        else:
            st.session_state.center_x = 0.3
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 90

    if country == "jap":
        st.session_state.center_x = 0
        st.session_state.center_y = 0
        st.session_state.angle_degrees = 0

    if country == "pal":
        st.session_state.center_x = -1.5
        st.session_state.center_y = 0
        st.session_state.angle_degrees = 0


def download_csv(url: str, destination: Union[str, Path]) -> None:
    """
    Downloads a CSV file from a specified URL and saves it to a given destination,
    displaying the download progress in a Streamlit app.

    Args:
        url (str): The URL of the CSV file to download.
        destination (Union[str, Path]): The local file path or Path object where the CSV file will be saved.

    Returns:
        None: This function does not return a value but writes the downloaded content to a file
              and updates the Streamlit UI with the download progress.

    Example:
        download_csv(
            "https://example.com/largefile.csv",
            "path/to/save/largefile.csv",
        )
    """
    # Send a GET request
    response = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kbyte
    progress_bar = st.progress(0)
    progress_status = st.empty()
    written = 0

    with open(destination, "wb") as f:
        for data in response.iter_content(block_size):
            written += len(data)
            f.write(data)
            # Update progress bar
            progress = int(100 * written / total_size)
            progress_bar.progress(progress)
            progress_status.text(f"> {progress}%")

    progress_status.text("Download complete.")
    progress_bar.empty()  # clear  the progress bar after completion


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
        start=(-1, -2),
        threshold=0.2,
    )
    _, interior = generate_oval_shape_points(
        num_points=50,
        radius=1.65 - 0.4,
        length=2,
        start=(-1, -1.2),
        threshold=0.2,
    )

    return exterior, interior


def sorting_key(filename: str) -> Tuple[int, str]:
    """
    Determine the sorting order of filenames based on their prefixes.

    The function assigns a tuple as a sorting key, where the first element is an integer
    representing the category of the file and the second element is the filename itself.
    The sorting categories are defined as follows:
    - Filenames starting with "female" are placed first (category 0).
    - Filenames starting with "male" are placed second (category 1).
    - Filenames starting with "mix_sorted" are placed third (category 2).
    - Filenames starting with "mix_random" are placed fourth (category 3).
    - Filenames that don't match any of the above categories are placed last (category 4).

    Parameters:
    - filename (str): The name of the file to be sorted.

    Returns:
    - Tuple[int, str]: A tuple containing the category and the filename, used for sorting.
    """
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
    """Rename columns of the dataframe based on the given mapping."""
    return data.rename(columns=mapping, inplace=True)


def set_column_types(data: pd.DataFrame, col_types: dict[str, Any]) -> pd.DataFrame:
    """Set the types of the dataframe columns based on the given column types."""
    # Ensure columns are in data before type casting
    valid_types = {
        col: dtype for col, dtype in col_types.items() if col in data.columns
    }
    return data.astype(valid_types)


def calculate_fps(data: pd.DataFrame) -> int:
    """Calculate fps based on the mean difference of the 'time' column."""
    mean_diff = data.groupby("id")["time"].diff().dropna().mean()
    return int(round(1 / mean_diff))


def load_file(file: str) -> pedpy.TrajectoryData:
    """Loads and processes a file to create a TrajectoryData object.

    This function reads a CSV file into a pandas DataFrame, renames columns according
    to a mapping provided in the session state, sets data types for the columns based
    on another mapping in the session state, calculates the frames per second (fps)
    from the data, and finally creates a TrajectoryData object with the processed data
    and the calculated fps.

    Parameters:
    - file (str): The path to the CSV file to be loaded.

    Returns:
    - An instance of TrajectoryData containing the processed data and frame rate.

    Note:
    - This function relies on global state (`st.session_state`) for configuration,
      which includes `rename_mapping` and `column_types`.
    - The `calculate_fps` function is assumed to calculate frames per second from the DataFrame.
    - The `TrajectoryData` class is assumed to be part of the `pedpy` module and requires
      the data DataFrame and the frame rate (fps) for initialization.
    """

    data = pd.read_csv(file)
    rename_columns(data, st.session_state.config.rename_mapping)
    set_column_types(data, st.session_state.config.column_types)
    fps = calculate_fps(data)
    trajectory_data = pedpy.TrajectoryData(data=data, frame_rate=fps)
    return trajectory_data


def rotate_trajectories(
    df: pd.DataFrame, shift_x: float, shift_y: float, angle_degrees: float
) -> pd.DataFrame:
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


def plot_neighbors_analysis(data, ids, exterior, interior, do_rotate):
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
    if do_rotate:
        rotated_data = rotate_trajectories(
            data,
            st.session_state.center_x,
            st.session_state.center_y,
            st.session_state.angle_degrees,
        )
    else:
        rotated_data = data
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
    # pos_neighbors = []

    # data_frame = rotated_data.loc[rotated_data["frame"] == frame]

    # pos_agent = data_frame.loc[data_frame["id"] == agent, ["x", "y"]].values.flatten()

    # st.dataframe(pos_agent)
    # neighbor_pos = data_frame.loc[data_frame["id"] == 3, ["x", "y"]].values.flatten()
    # dist = np.linalg.norm(pos_agent - neighbor_pos)
    # pos_neighbors.append(dist)
    # st.info(f"{dist}")
    # st.dataframe(neighbor_pos)

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


def get_numbers_country(country: str) -> Tuple[int, int, int, int]:
    """
    Count the number of files by category for a specified country within the session state configuration.

    This function iterates over a list of files configured in the session state for the given country.
    It categorizes the files based on their naming conventions into four categories: female, male,
    mix_sorted, and mix_random. The function then counts the number of files in each category.

    Parameters:
    - country (str): The name of the country for which to count the files.

    Returns:
    - Tuple[int, int, int, int]: A tuple containing the counts of files in the following order:
        n_female: Number of files prefixed with "female".
        n_male: Number of files prefixed with "male".
        n_mixed_random: Number of files prefixed with "mix_random".
        n_mixed_sorted: Number of files prefixed with "mix_sorted".

    Note:
    - The function assumes that `st.session_state.config.files` is a dictionary where each key is a country name,
      and the corresponding value is a list of file paths.
    - Files are categorized based on prefixes in their names.
    """

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


def set_rotation_variables(selected_file, country):
    ger_substrings_to_check = [
        "female",
        "20_19",
        "16_18",
        "40_11",
        "8_16",
        "8_17",
        "32_11",
        "36_11",
        "4_11",
        "4_12",
        "4_13",
        "4_15",
        "4_14",
    ]
    aus_substrings_to_check = ["female", "mix_sorted_40_01", "mix_random_41_01"]
    if country == "ger":
        if any(substring in selected_file for substring in ger_substrings_to_check):
            st.session_state.center_x = 3.1
            st.session_state.center_y = 3
            st.session_state.angle_degrees = 90
        else:
            st.session_state.center_x = 3.1
            st.session_state.center_y = -3
            st.session_state.angle_degrees = 90

    if country == "aus":
        if any(substring in selected_file for substring in aus_substrings_to_check):
            st.session_state.center_x = 1.7
            st.session_state.center_y = -0.2
            st.session_state.angle_degrees = 85
        else:
            st.session_state.center_x = 1.8
            st.session_state.center_y = -6.3
            st.session_state.angle_degrees = 89

    if country == "chn":
        if "female" in selected_file:
            st.session_state.center_x = 0.1
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 87
        elif "mix_random" in selected_file:
            st.session_state.center_x = 0.1
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 85
        else:
            st.session_state.center_x = 0.3
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 90

    if country == "jap":
        st.session_state.center_x = 0
        st.session_state.center_y = 0
        st.session_state.angle_degrees = 0

    if country == "pal":
        st.session_state.center_x = -1.5
        st.session_state.center_y = 0
        st.session_state.angle_degrees = 0


def show_fig(fig: Figure, html: bool = False, height: int = 500) -> None:
    """Workaround function to show figures having LaTeX-Code.

    Args:
        fig (Figure): A Plotly figure object to display.
        html (bool, optional): Flag to determine if the figure should be shown as HTML. Defaults to False.
        height (int, optional): Height of the HTML component if displayed as HTML. Defaults to 500.

    Returns:
        None
    """
    if not html:
        st.plotly_chart(fig)
    else:
        st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=height)
