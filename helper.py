import numpy as np
from typing import Tuple, Any
import pandas as pd
import streamlit as st
import pedpy


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
