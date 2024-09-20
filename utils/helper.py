"""Collection of some helpful functions."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pedpy
import plotly.graph_objects as go
import requests  # type: ignore
import streamlit as st
from plotly.graph_objs import Figure
from shapely.geometry import Polygon
from visualization import plots as pl
import streamlit.components.v1 as components

Point: TypeAlias = Tuple[float, float]
st_column: TypeAlias = st.delta_generator.DeltaGenerator


def is_running_locally() -> bool:
    """Check if the Streamlit app is running locally."""
    streamlit_server = "/mount/src/gender-experiments"
    current_working_directory = os.getcwd()
    return current_working_directory != streamlit_server


def zip_directory(path_to_directory: Path) -> str:
    """Zips the specified directory and returns the path to the zipped file."""
    output_filename = f"{path_to_directory}"
    shutil.make_archive(output_filename, "zip", path_to_directory)

    return f"{output_filename}.zip"


def download_zipped_directory(zipped_path: str) -> None:
    """Create a download button for the zipped file."""
    with open(zipped_path, "rb") as file:
        st.download_button(
            label="Download ZIP",
            data=file,
            file_name=os.path.basename(zipped_path),
            mime="application/zip",
        )


def download_csv(url: str, destination: Union[str, Path]) -> None:
    """
    Download a CSV file from a specified URL and saves it to a given destination.

    Display the download progress in a Streamlit app.

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
    logging.info(f"requesting from {url} ...")
    # Total size in bytes.
    total_size = int(response.headers.get("content-length", 0))
    logging.info(f"Got <{total_size}>.")
    if total_size == 0:
        st.error(f"Could not download file from {url}.")
        st.stop()

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


def dist(p1: Point, p2: Point) -> float:
    """Calculate the Euclidean distance between two points.

    Parameters:
    - point1: A tuple representing the coordinates of the first point.
    - point2: A tuple representing the coordinates of the second point.

    Returns:
    - The Euclidean distance between point1 and point2.
    """
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def append_point_if_threshold_met(
    points: List[Point],
    selected_points: List[Point],
    tmp_point: Point,
    last_selected: Point,
    threshold: float,
) -> Tuple[List[Point], List[Point], Point]:
    """
    Append a point to the list of points and selected points if it meets the distance threshold from the last selected point.

    Parameters:
    - points: The current list of points.
    - selected_points: The current list of selected points.
    - tmp_point: The point to potentially add.
    - last_selected: The last point that was added to the selected points.
    - threshold: The minimum distance required to add the point to the selected points.

    Returns:
    - Updated lists of points and selected points, and the last selected point.
    """
    points.append(tmp_point)
    if dist(tmp_point, last_selected) >= threshold:
        selected_points.append(tmp_point)
        last_selected = tmp_point
    return points, selected_points, last_selected


def generate_oval_shape_points(
    num_points: int,
    length: float = 2.3,
    radius: float = 1.65,
    start: Tuple[float, float] = (0.0, 0.0),
    dx: float = 0.2,
    threshold: float = 0.5,
) -> Tuple[List[Point], List[Point]]:
    """
    Generate points along a linear segment.

    Parameters:
    - start: The starting point of the segment.
    - dx: The increment in the x-direction for each point.
    - length: The total length of the segment.
    - radius: The radius used to adjust the y-coordinate for the second segment.
    - threshold: The threshold distance for selecting points.
    - is_second_segment: Flag to adjust points for the second segment.

    Returns:
    - A list of points along the segment.
    """
    points: List[Point] = [start]
    selected_points: List[Point] = [start]
    last_selected: Point = start

    dphi = 0.005 / radius
    center1 = (start[0], start[1] + radius)
    center2 = (start[0] + length, start[1] + radius)

    # Generate points for the first segment
    segment_points = generate_segment_points(start, dx, length, radius, threshold)
    for p in segment_points:
        points, selected_points, last_selected = append_point_if_threshold_met(points, selected_points, p, last_selected, threshold)

    # Generate points for the first half circle
    circle_points = generate_half_circle_points(center2, radius, dphi, -np.pi / 2, np.pi / 2)
    for p in circle_points:
        points, selected_points, last_selected = append_point_if_threshold_met(points, selected_points, p, last_selected, threshold)

    # Generate points for the second segment
    segment_points = generate_segment_points(start, dx, length, radius, threshold, is_second_segment=True)
    for p in segment_points:
        points, selected_points, last_selected = append_point_if_threshold_met(points, selected_points, p, last_selected, threshold)

    # Generate points for the second half circle
    circle_points = generate_half_circle_points(center1, radius, dphi, np.pi / 2, 3 * np.pi / 2 - dphi)
    for p in circle_points:
        points, selected_points, last_selected = append_point_if_threshold_met(points, selected_points, p, last_selected, threshold)

    # Final adjustments
    if dist(selected_points[-1], start) < threshold:
        selected_points.pop()
    if num_points > len(selected_points):
        print(f"Warning: Requested {num_points} points, but only {len(selected_points)} can be provided.")

    selected_points = selected_points[:num_points]
    return points, selected_points


def generate_segment_points(
    start: Point,
    dx: float,
    length: float,
    radius: float,
    threshold: float,
    is_second_segment: bool = False,
) -> List[Point]:
    """
    Generate points along a linear segment.

    Parameters:
    - start: The starting point of the segment.
    - dx: The increment in the x-direction for each point.
    - length: The total length of the segment.
    - radius: The radius used to adjust the y-coordinate for the second segment.
    - threshold: The threshold distance for selecting points.
    - is_second_segment: Flag to adjust points for the second segment.

    Returns:
    - A list of points along the segment.
    """
    segment_points = []
    npoint_on_segment = int(length / dx)
    for i in range(1, npoint_on_segment + 1):
        if is_second_segment:
            tmp_point = (
                start[0] + (npoint_on_segment + 1) * dx - i * dx,
                start[1] + 2 * radius,
            )
        else:
            tmp_point = (start[0] + i * dx, start[1])
        segment_points.append(tmp_point)
    return segment_points


def generate_half_circle_points(
    center: Point,
    radius: float,
    dphi: float,
    start_phi: float,
    end_phi: float,
) -> List[Point]:
    """
    Generate points along a half-circle.

    Parameters:
    - center: The center point of the half-circle.
    - radius: The radius of the half-circle.
    - dphi: The increment in the angle (in radians) for generating points.
    - start_phi: The starting angle (in radians) for generating points.
    - end_phi: The ending angle (in radians) for generating points.

    Returns:
    - A list of points along the half-circle.
    """
    circle_points = []
    for phi in np.arange(start_phi, end_phi, dphi):
        x = center[0] + radius * np.cos(phi)
        y = center[1] + radius * np.sin(phi)
        circle_points.append((x, y))
    return circle_points


# Function to calculate the length of a closed parcours
def parcours_length(points: Polygon) -> float:
    # Initialize total length
    total_length = 0

    # Iterate through the points and calculate distances between consecutive points
    for i in range(len(points)):
        # Get the current point and the next point (loop back to the first point at the end)
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]  # Next point, wrapping around

        # Calculate the Euclidean distance between two points
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Add the distance to the total length
        total_length += distance

    return total_length


def generate_parcour() -> Tuple[Polygon, Polygon, Polygon]:
    _, exterior = generate_oval_shape_points(
        num_points=100,
        radius=1.65 + 0.4,
        length=2.3,
        start=(-1, -2),
        threshold=0.2,
    )

    _, interior = generate_oval_shape_points(
        num_points=50,
        radius=1.65 - 0.4,
        length=2.3,
        start=(-1, -1.2),
        threshold=0.2,
    )

    _, middle_path = generate_oval_shape_points(
        num_points=250,
        radius=1.65,
        length=2,
        start=(-1, -1.6),
        dx=0.05,
        threshold=0.05,
    )
    # print("ext", parcours_length(exterior))
    # print("int", parcours_length(interior))
    # print("middle", parcours_length(middle_path))

    return exterior, interior, middle_path


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
    if filename.startswith("male"):
        return (1, filename)
    if filename.startswith("mix_sorted"):
        return (2, filename)
    if filename.startswith("mix_random"):
        return (3, filename)

    return (4, filename)  # For filenames that don't match any category


def rename_columns(data: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns of the dataframe based on the given mapping."""
    return data.rename(columns=mapping, inplace=True)


def set_column_types(data: pd.DataFrame, col_types: dict[str, Any]) -> pd.DataFrame:
    """Set the types of the dataframe columns based on the given column types."""
    # Ensure columns are in data before type casting
    valid_types = {col: dtype for col, dtype in col_types.items() if col in data.columns}
    return data.astype(valid_types)


def calculate_fps(data: pd.DataFrame) -> int:
    """Calculate fps based on the mean difference of the 'time' column."""
    mean_diff = data.groupby("id")["time"].diff().dropna().mean()
    return int(round(1 / mean_diff))


def load_file(file: str, sep: str = ",") -> pedpy.TrajectoryData:
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

    data = pd.read_csv(file, sep=sep)
    rename_columns(data, st.session_state.config.rename_mapping)
    set_column_types(data, st.session_state.config.column_types)
    fps = calculate_fps(data)
    return pedpy.TrajectoryData(data=data, frame_rate=fps)


def get_neighbors_special_agent_data(
    agent: int,
    frame: int,
    df: pd.DataFrame,
    nearest_dist: npt.NDArray[np.float64],
    nearest_ind: npt.NDArray[np.int_],
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    float,
    npt.NDArray[np.float64],
    List[str],
]:
    # Filter DataFrame for the specified fram
    frames = df["frame"].to_numpy()
    first_frame = frames[0]
    at_frame = df[(df["frame"] == frame)]

    # Extract points, speeds, and ids
    points = at_frame[["x", "y"]].to_numpy()
    point_agent = df[(df["frame"] == frame) & (df["id"] == agent)][["x", "y"]].to_numpy()
    point_agent_future = df[(df["frame"] == frame + 50) & (df["id"] == agent)][["x", "y"]].to_numpy()

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
        return np.array([]), np.array([]), 0, np.array([]), []

    if frame == first_frame:
        distance_now = (neighbors[0][0] - point_agent[0][0]) ** 2 + (neighbors[0][1] - point_agent[0][1]) ** 2
        distance_future = (neighbors[0][0] - point_agent_future[0][0]) ** 2 + (neighbors[0][1] - point_agent_future[0][1]) ** 2

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


def save_changes(selected_file: str, data: pd.DataFrame) -> None:
    """Save changes to file."""
    path_parts = Path(selected_file).parts
    original_directory = Path(*path_parts[:-1])
    directory = Path(str(original_directory) + "_enhanced")
    filename = Path(path_parts[-1])
    st.info(directory)
    directory.mkdir(parents=True, exist_ok=True)
    newfile = directory / filename
    st.warning(newfile)
    rename_mapping = {
        "id": "ID",
        "time": "t(s)",
        "x": "x(m)",
        "y": "y(m)",
    }
    wdata = data.copy()
    wdata.rename(columns=rename_mapping, inplace=True)
    selected_columns = [
        "ID",
        "next",
        "prev",
        "gender",
        "frame",
        "t(s)",
        "x(m)",
        "y(m)",
    ]

    wdata[selected_columns].to_csv(newfile, index=False)
    zip_file = zip_directory(directory)
    download_zipped_directory(zip_file)


def update_row(
    next_: int,
    next0: int,
    prev0: int,
    prev: int,
    data: pd.DataFrame,
    agent: int,
    add: bool,
) -> pd.DataFrame:
    """Write changed row in data frame."""
    correct = (next_ != next0) or (prev0 != prev)
    if next_ != next0:
        what_changed = next_
    elif prev0 != prev:
        what_changed = prev
    else:
        what_changed = -1
    if correct or add:
        if add:
            st.info(f"added next {next_}, prev {prev}.")
        else:
            st.info(f"added {what_changed}.")
        data.loc[data["id"] == agent, ["prev", "next"]] = [prev, next_]
        st.divider()

    return data


def handle_prev_next_neighbors(data: pd.DataFrame, frame: int, next_: int, prev: int, ids: npt.NDArray[np.int_]) -> Tuple[npt.NDArray[Any], List[int], List[str], int, int]:
    """Handle neighborhood since some data are not circular."""
    pos_next = data.loc[(data["frame"] == frame) & (data["id"] == next_), ["x", "y"]].values
    pos_prev = data.loc[(data["frame"] == frame) & (data["id"] == prev), ["x", "y"]].values
    if pos_next.any():
        pos_next = pos_next[0]

    if pos_prev.any():
        pos_prev = pos_prev[0]

    if next_ < min(ids):
        next_ = np.nan  # type: ignore
    if prev < min(ids):
        prev = np.nan  # type: ignore

    neighbors_ids = [next_, prev]
    neighbor_type = ["next", "prev"]
    if np.isnan(next_):
        neighbors_ids = [prev]
        neighbor_type = ["prev"]
    if np.isnan(prev):
        neighbors_ids = [next_]
        neighbor_type = ["next"]

    if pos_next.size > 0 and pos_prev.size > 0:
        neighbors = np.vstack((pos_next, pos_prev))
    elif pos_next.size > 0:
        neighbors = pos_next.reshape(-1, pos_next.shape[-1])
        neighbors_ids = [next_]
        neighbor_type = ["next"]
    else:
        neighbors = pos_prev.reshape(-1, pos_prev.shape[-1])
        neighbors_ids = [prev]
        neighbor_type = ["prev"]
    return neighbors, neighbors_ids, neighbor_type, next_, prev


def init_neighbors(data: pd.DataFrame, frames: npt.NDArray[np.int_]) -> pd.DataFrame:
    """Initialize neighbors for each agent in the dataset."""
    st.info("Initializing neighbors...")
    for fr in frames:
        data0 = data[data["frame"] == fr].copy()
        data0_sorted = data0.sort_values(by="x")
        data0_sorted.reset_index(drop=True, inplace=True)
        sorted_ids = data0_sorted["id"].tolist()
        for index, current_id in enumerate(sorted_ids):
            prev_id = sorted_ids[index - 1] if index > 0 else np.nan
            next_id = sorted_ids[index + 1] if index < len(sorted_ids) - 1 else np.nan
            data.loc[data["id"] == current_id, "prev"] = prev_id
            data.loc[data["id"] == current_id, "next"] = next_id
    st.info("Done initializing neighbors.")
    return data


def handle_user_input(
    n0: st_column,
    n1: st_column,
    n3: st_column,
    frames: pd.DataFrame,
    prev0: int,
    next0: int,
) -> Tuple[int, int, int]:
    """Return frame, prev and next for correction."""
    frame = int(
        n3.number_input(
            "Frame",
            int(frames[0]),
            int(np.max(frames)),
            int(frames[0]),
            step=10,
            help="Frame at which to display a snapshot of the neighbors",
        )
    )

    prev = int(
        n0.number_input(
            "prev",
            value=prev0,
            format="%d",
        )
    )

    next_ = int(
        n1.number_input(
            "next",
            value=next0,
            format="%d",
        )
    )

    return frame, next_, prev


def initialise_prev_next(
    is_prev_in_df: bool,
    is_next_in_df: bool,
    data: pd.DataFrame,
    agent: int,
    frames: npt.NDArray[np.int_],
) -> Tuple[int, int, pd.DataFrame]:
    next0 = -1
    prev0 = -1
    if is_prev_in_df and is_next_in_df:
        neighbors_tmp = data.loc[data["id"] == agent, ["next", "prev"]].iloc[0].values
        if not np.isnan(neighbors_tmp[0]):
            next0 = int(neighbors_tmp[0])
        if not np.isnan(neighbors_tmp[1]):
            prev0 = int(neighbors_tmp[1])
    else:
        data[["next", "prev"]] = np.nan

        data = init_neighbors(data, frames)
        neighbors_tmp = data.loc[data["id"] == agent, ["next", "prev"]].iloc[0].values
        if not np.isnan(neighbors_tmp[0]):
            next0 = int(neighbors_tmp[0])
        if not np.isnan(neighbors_tmp[1]):
            prev0 = int(neighbors_tmp[1])

    return prev0, next0, data


def plot_neighbors_analysis(
    selected_file: str,
    data: pd.DataFrame,
    ids: npt.NDArray[np.int_],
    exterior: Polygon,
    interior: Polygon,
    middle_path: Polygon,
) -> Tuple[go.Figure, pd.DataFrame]:
    # st.dataframe(data)

    n0, n1, n2, n3 = st.columns((1, 1, 1, 1))
    agent = int(
        n2.number_input(
            "Agent",
            min_value=int(min(ids)),
            max_value=int(max(ids)),
            value=int(min(ids)),
            placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
            format="%d",
        )
    )

    frames = data["frame"].to_numpy()
    is_prev_in_df = "prev" in data.columns
    is_next_in_df = "next" in data.columns

    prev0, next0, data = initialise_prev_next(is_prev_in_df, is_next_in_df, data, agent, frames)
    frame, next_, prev = handle_user_input(n0, n1, n3, frames, prev0, next0)
    neighbors, neighbors_ids, neighbor_type, next_, prev = handle_prev_next_neighbors(data, frame, next_, prev, ids)

    fig = pl.plot_agent_and_neighbors(
        agent,
        frame,
        data,
        neighbors,
        neighbors_ids,
        exterior,
        interior,
        middle_path,
        neighbor_type,
    )
    write_to_file = n0.button("Write to file")
    if write_to_file:
        save_changes(selected_file, data)

    add = n1.button("Force add line to data")
    data = update_row(next_, next0, prev0, prev, data, agent, add)

    return fig, data


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
        components.html(fig.to_html(include_mathjax="cdn"), height=height)


def increment_page_start(page_size: int) -> None:
    """Implement pagination to show large dataframe."""
    st.session_state.page_start += page_size


def decrement_page_start(page_size: int) -> None:
    """Implement pagination to show large dataframe."""
    st.session_state.page_start -= page_size


def project_position_to_path(position: npt.NDArray[np.float64], path: Polygon) -> Tuple[np.int_, np.float64]:
    """Project a position onto the path by finding the point with the minimum Δx and Δy differences."""
    delta_sum = [np.linalg.norm(position - p) for p in path]
    closest_point_index = np.argmin(delta_sum)
    return closest_point_index, delta_sum[closest_point_index]


def precompute_path_distances(path: Polygon) -> npt.NDArray[np.float64]:
    """Eucleadian distance."""
    return np.array([np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1)])


def sum_distances_between_agents_on_path(
    agent1_pos: npt.NDArray[np.float64],
    agent2_pos: npt.NDArray[np.float64],
    path: Polygon,
    path_distances: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the distance between two agents by summing the distances between points on the path that lie between them."""
    index1, _ = project_position_to_path(agent1_pos, path)
    index2, _ = project_position_to_path(agent2_pos, path)
    np.linalg.norm(agent1_pos - path[index1])
    np.linalg.norm(agent2_pos - path[index2])
    p1, p2 = path[index1], path[index2]
    # Ensure index1 is smaller than index2 for simplicity
    if index1 > index2:
        index1, index2 = index2, index1
    direct_distance_sum = np.sum(path_distances[index1:index2])
    loop_around_distance_sum = np.sum(path_distances[index2:]) + np.sum(path_distances[:index1])
    # Choose the shorter distance
    distance_sum = min(direct_distance_sum, loop_around_distance_sum)
    return distance_sum, p1, p2
