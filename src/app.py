# import numpy as np
import glob
import time
from collections import defaultdict

# from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from shapely import Polygon
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import pedpy
import plotly.graph_objects as go
import streamlit as st


@dataclass
class DataConfig:
    rename_mapping: dict
    column_types: dict
    countries: list
    files: dict = field(default_factory=dict)
    data: dict = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        self.retrieve_files()

    def retrieve_files(self):
        for country in self.countries:
            self.files[country] = glob.glob(f"../{country}/*.csv") + glob.glob(
                f"../{country}/*.txt"
            )


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


import pandas as pd
import numpy as np


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


# @st.cache_data
def plot_trajectories(
    data: pd.DataFrame, framerate: int, uid: int, exterior, interior
) -> go.Figure:
    fig = go.Figure()
    rotated = st.checkbox("Rotated", value=True)
    gender_colors = {
        1: "blue",  # Assuming 1 is for female
        2: "green",  # Assuming 2 is for male
        0: "red",  # non binary
    }
    x_exterior, y_exterior = Polygon(exterior).exterior.xy
    x_exterior = list(x_exterior)
    y_exterior = list(y_exterior)
    x_interior, y_interior = Polygon(interior).exterior.xy
    x_interior = list(x_interior)
    y_interior = list(y_interior)

    rotated_data = rotate_trajectories(
        data,
        st.session_state.center_x,
        st.session_state.center_y,
        st.session_state.angle_degrees,
    )
    # For each unique id, plot a trajectory
    if uid is not None:
        df = data[data["id"] == uid]
        color_choice = gender_colors[df["gender"].iloc[0]]
        if not rotated:
            fig.add_trace(
                go.Scatter(
                    x=df["x"][::framerate],
                    y=df["y"][::framerate],
                    line=dict(color=color_choice),
                    marker=dict(color=color_choice),
                    mode="lines",
                    name=f"ID {uid}",
                )
            )
        rotated_df = rotated_data[rotated_data["id"] == uid]
        if rotated:
            fig.add_trace(
                go.Scatter(
                    x=rotated_df["x"][::framerate],
                    y=rotated_df["y"][::framerate],
                    line=dict(color=color_choice, dash="dash"),
                    marker=dict(color=color_choice),
                    mode="lines",
                    name=f"ID {uid}",
                )
            )
    else:
        for uid, df in data.groupby("id"):
            color_choice = gender_colors[df["gender"].iloc[0]]
            if not rotated:
                fig.add_trace(
                    go.Scatter(
                        x=df["x"][::framerate],
                        y=df["y"][::framerate],
                        line=dict(color=color_choice),
                        marker=dict(color=color_choice),
                        mode="lines",
                        name=f"ID {uid}",
                    )
                )
        if rotated:
            for uid, rotated_df in rotated_data.groupby("id"):
                fig.add_trace(
                    go.Scatter(
                        x=rotated_df["x"][::framerate],
                        y=rotated_df["y"][::framerate],
                        line=dict(color=color_choice, dash="dash"),
                        marker=dict(color=color_choice),
                        mode="lines",
                        name=f"ID {uid}",
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=x_exterior,
            y=y_exterior,
            mode="lines",
            line=dict(color="black"),
            name="exterior",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_interior,
            y=y_interior,
            mode="lines",
            line=dict(color="black"),
            name="interior",
        )
    )
    fig.update_layout(
        title="Trajectories",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y"),
        yaxis=dict(scaleratio=1),
        showlegend=False,
    )
    return fig


# Main
msg = st.empty()
if not hasattr(st.session_state, "loaded_data"):
    st.session_state.loaded_data = {}

if "rotate" not in st.session_state:
    st.session_state["rotate"] = False
    st.session_state["center_x"] = 0.0
    st.session_state["center_y"] = 0.0
    st.session_state["angle_degrees"] = 90.0

if not hasattr(st.session_state, "config"):
    msg.info("init config")
    st.session_state.config = DataConfig(
        rename_mapping={
            "ID": "id",
            "t(s)": "time",
            "x(m)": "x",
            "y(m)": "y",
        },
        column_types={"id": int, "gender": int, "time": float, "x": float, "y": float},
        countries=["aus", "chn", "ger", "jap", "pal"],
        # countries = ["ger"]
    )

if not hasattr(st.session_state, "figures"):
    st.session_state.figures = {}

for country in st.session_state.config.countries:
    if country not in st.session_state.loaded_data:
        files = st.session_state.config.files[country]
        if not files:
            msg.warning(f"{country}: data missing")
            continue

        st.write(f"Processing data for <{country}>")
        progress_text = st.empty()
        progress = st.progress(0)
        num_files = len(files)
        st.session_state.loaded_data[country] = []

        for idx, file in enumerate(files):
            data = pd.read_csv(file)
            rename_columns(data, st.session_state.config.rename_mapping)
            set_column_types(data, st.session_state.config.column_types)
            if file == files[0]:
                fps = calculate_fps(data)

            trajectory_data = pedpy.TrajectoryData(data=data, frame_rate=fps)
            st.session_state.loaded_data[country].append(trajectory_data)

            # Update the progress bar
            progress_value = (idx + 1) / num_files
            progress.progress(progress_value)
            progress_text.text(f"File {idx + 1} of {num_files}")


st.title("Trajectory Visualization")
c1, c2 = st.columns((1, 1))
country = c1.selectbox("Select a country:", st.session_state.config.countries)
msg.write("")
if country:
    files = st.session_state.config.files[country]
    selected_file = c2.selectbox("Select a file:", files)
    if selected_file:
        file_index = files.index(selected_file)
        # default values
        if country == "ger":
            if "female" in selected_file:
                st.session_state.center_x = 3.1
                st.session_state.center_y = 3
                st.session_state.angle_degrees = 90
            else:
                st.session_state.center_x = 3.1
                st.session_state.center_y = -3
                st.session_state.angle_degrees = 90

        if country == "aus":
            if "female" in selected_file:
                st.session_state.center_x = 1.7
                st.session_state.center_y = 0
                st.session_state.angle_degrees = 90
            else:
                st.session_state.center_x = 1.7
                st.session_state.center_y = -6.3
                st.session_state.angle_degrees = 90

        if country == "chn":
            if "female" in selected_file:
                st.session_state.center_x = 0.1
                st.session_state.center_y = 0
                st.session_state.angle_degrees = 90
            else:
                st.session_state.center_x = 0.1
                st.session_state.center_y = 0
                st.session_state.angle_degrees = 90

        if country == "pal":
         
            st.session_state.center_x = -1.5
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 0

        trajectory_data = st.session_state.loaded_data[country][file_index]
        data = trajectory_data.data
        start_time = time.time()
        #        if selected_file not in st.session_state.figures.keys():
        framerate = st.slider("Every nth frame", 1, 100, 80, 10)
        ids = data["id"].unique()
        uid = st.number_input(
            "Insert id of pedestrian",
            value=None,
            min_value=int(min(ids)),
            max_value=int(max(ids)),
            placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
            format="%d",
        )
        rc1, rc2, rc3 = st.columns((1, 1, 1))
        center_x = rc1.number_input(
            "Shift X:",
            value=float(st.session_state["center_x"]),
            step=0.1,
            help="AF=1.7, AM=1.7, CN=0.1, G=3, P=-1.5",
        )
        center_y = rc2.number_input(
            "Shift Y:",
            value=float(st.session_state["center_y"]),
            step=0.1,
            help="AF=0, AM=6.3, CN=0",
        )
        angle_degrees = rc3.number_input(
            "Angle in Degrees:", value=st.session_state["angle_degrees"],
            help="A=90, CN=90, G=3"
        )
        if center_x != st.session_state.center_x:
            st.session_state.center_x = center_x

        if center_y != st.session_state.center_y:
            st.session_state.center_y = center_y

        if angle_degrees != st.session_state.angle_degrees:
            st.session_state.angle_degrees = angle_degrees

        _, exterior = generate_oval_shape_points(
            num_points=50, radius=1.65 + 0.4, start=(-1.0, -2), threshold=0.2
        )
        _, interior = generate_oval_shape_points(
            num_points=50, radius=1.65 - 0.4, start=(-1, -1.2), threshold=0.2
        )
        fig = plot_trajectories(data, framerate, uid, exterior, interior)
        st.plotly_chart(fig)
        # if st.session_state['rotate']:
        #    # Rotate and display the dataframe
        #    rotated_df = rotate_trajectories(data, center_x, center_y, angle_degrees)

        # fig2 = plot_trajectories(rotated_df, framerate, uid, exterior, interior)
        # st.plotly_chart(fig2)

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Time taken to plot trajectories: {elapsed_time:.2f} seconds")
