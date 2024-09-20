"""Collection of ploting functionalities."""

from typing import Any, Dict, List, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import spatial
from shapely import Polygon
import matplotlib.pyplot as plt
import src.utils.helper as hp
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

st_column: TypeAlias = st.delta_generator.DeltaGenerator


def plot_trajectories(
    data: pd.DataFrame,
    framerate: int,
    uid: Union[int | float | None],
    exterior: Polygon,
    interior: Polygon,
    plot_parcour: bool,
) -> go.Figure:
    """Plot trajectories."""
    fig = go.Figure()
    c1, c2, c3 = st.columns((1, 1, 1))
    num_agents = len(np.unique(data["id"]))
    male_agents = data[data["gender"] == 2]
    female_agents = data[data["gender"] == 1]
    unknown_agents = data[data["gender"].isin([0, -1])]

    # Count unique IDs in each subset
    num_unique_males = male_agents["id"].nunique()
    num_unique_females = female_agents["id"].nunique()
    num_unique_unknowns = unknown_agents["id"].nunique()

    rotated = False  # c1.checkbox("Rotate and shift", value=False)

    gender_map = {1: "F", 2: "M", 0: "N", -1: "E"}
    gender_colors = {
        1: "blue",  # Assuming 1 is for female
        2: "green",  # Assuming 2 is for male
        0: "black",  # non binary
        -1: "yellow",
    }
    x_exterior, y_exterior = Polygon(exterior).exterior.xy
    x_exterior = list(x_exterior)
    y_exterior = list(y_exterior)
    x_interior, y_interior = Polygon(interior).exterior.xy
    x_interior = list(x_interior)
    y_interior = list(y_interior)
    rotated_data = data
    # For each unique id, plot a trajectory
    if uid is not None:
        df = data[data["id"] == uid]
        gender = gender_map[df["gender"].iloc[0]]
        color_choice = gender_colors[df["gender"].iloc[0]]
        if not rotated:
            fig.add_trace(
                go.Scatter(
                    x=df["x"][::framerate],
                    y=df["y"][::framerate],
                    line={"color": color_choice},
                    marker={"color": color_choice},
                    mode="lines",
                    name=f"ID {uid}, {gender}",
                )
            )
        rotated_df = rotated_data[rotated_data["id"] == uid]
        if rotated:
            color_choice = gender_colors[rotated_df["gender"].iloc[0]]
            fig.add_trace(
                go.Scatter(
                    x=rotated_df["x"][::framerate],
                    y=rotated_df["y"][::framerate],
                    line={"color": color_choice},
                    marker={"color": color_choice},
                    mode="lines",
                    name=f"ID {uid}, {gender}",
                )
            )
    else:
        for uid, df in data.groupby("id"):
            color_choice = gender_colors[df["gender"].iloc[0]]
            gender = gender_map[df["gender"].iloc[0]]
            if not rotated:
                fig.add_trace(
                    go.Scatter(
                        x=df["x"][::framerate],
                        y=df["y"][::framerate],
                        line={"color": color_choice},
                        marker={"color": color_choice},
                        mode="lines",
                        name=f"ID {uid}, {gender}",
                    )
                )
        if rotated:
            for uid, rotated_df in rotated_data.groupby("id"):
                color_choice = gender_colors[rotated_df["gender"].iloc[0]]
                gender = gender_map[rotated_df["gender"].iloc[0]]
                fig.add_trace(
                    go.Scatter(
                        x=rotated_df["x"][::framerate],
                        y=rotated_df["y"][::framerate],
                        line={"color": color_choice},
                        marker={"color": color_choice},
                        mode="lines",
                        name=f"ID {uid}, {gender}",
                    )
                )

    if plot_parcour:
        fig.add_trace(
            go.Scatter(
                x=x_exterior,
                y=y_exterior,
                mode="lines",
                line={"color": "red"},
                name="exterior",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_interior,
                y=y_interior,
                mode="lines",
                line={"color": "red"},
                name="interior",
            )
        )
    np.min(x_exterior)
    np.max(x_exterior)
    np.min(y_exterior) - 0.5
    np.max(y_exterior) + 0.5
    fig.update_layout(
        title=f" Trajectories: {num_agents} | M: {num_unique_males} | F: {num_unique_females} | N: {num_unique_unknowns}",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis={"scaleanchor": "y"},  # , range=[xmin, xmax]),
        yaxis={"scaleratio": 1},  # , range=[ymin, ymax]),
        showlegend=False,
    )

    fig.add_annotation(
        x=0.5,
        y=1.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        text="<span style='color:green;'>Green: Male</span>, <span style='color:blue;'>Blue: Female</span>",
        font={"size": 12},
        align="center",
    )
    return fig


def plot_trajectories_matplotlib(data: pd.DataFrame, framerate: int, exterior: Polygon, interior: Polygon, figname: str, plot_parcour: bool) -> Figure:
    """Plot trajectories using Matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 5))

    gender_map = {1: "Female", 2: "Male"}  # , 0: "N", -1: "E"}
    gender_colors = {
        1: "blue",  # Assuming 1 is for female
        2: "green",  # Assuming 2 is for male
        0: "black",  # non binary
        -1: "yellow",
    }

    # Plot exterior and interior polygons
    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in gender_colors.values()]
    if plot_parcour:
        x_exterior, y_exterior = Polygon(exterior).exterior.xy
        x_interior, y_interior = Polygon(interior).exterior.xy
        ax.plot(x_exterior, y_exterior, color="black", label="Exterior")
        ax.plot(x_interior, y_interior, color="black", label="Interior")
        ax.legend(custom_lines, [f"{desc}" for gender, desc in gender_map.items()], loc="upper center", bbox_to_anchor=(0.5, 1.130), ncol=2)
    else:
        ax.legend(custom_lines, [f"{desc}" for gender, desc in gender_map.items()], loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    ax.set_xlabel(r"$x\; /\;m$", fontsize=16)
    ax.set_ylabel(r"$y\; /\;m$", fontsize=16)
    ax.set_aspect("equal", "box")

    # Plot each unique ID's trajectory
    for uid, group_df in data.groupby("id"):
        color_choice = gender_colors[group_df["gender"].iloc[0]]
        gender = gender_map[group_df["gender"].iloc[0]]
        ax.plot(group_df["x"][::framerate], group_df["y"][::framerate], lw=0.09, alpha=0.4, color=color_choice, label=f"ID {uid}, {gender}")

    # Customize plot
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Create custom legends for gender colors
    print(figname)
    plt.savefig(figname, bbox_inches="tight", pad_inches=0.0)
    plt.tight_layout()
    return fig


def plot_time_series(data: pd.DataFrame, speed: pd.DataFrame, fps: int, key_density: str) -> go.Figure:
    """Plot time series of density and frame side by side."""
    density = data[key_density]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Mean density: {np.mean(density):.2f} (+- {np.std(density):.2f}) 1/m/m",
            f"Mean speed: {np.mean(speed):.2f} (+- {np.std(speed):.2f}) / m/s",
        ),
    )
    # st.dataframe(data)
    if key_density == "individual_density":
        data = data.sort_values(by="frame")

    fig.add_trace(
        go.Scatter(
            x=data["frame"] / fps,
            y=density,
            line={"color": "blue"},
            marker={"color": "blue"},
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=speed.index / fps,
            y=speed,
            line={"color": "blue"},
            marker={"color": "blue"},
            mode="lines",
        ),
        row=1,
        col=2,
    )

    rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    rmax = 5  # np.max(data["instantaneous_density"]) + 0.5
    vmax = np.max(speed) + 0.5
    fig.update_layout(
        xaxis_title="Time / s",
        showlegend=False,
    )
    fig.update_yaxes(range=[rmin, rmax], title_text="Density / 1/m/m", row=1, col=1)
    fig.update_yaxes(range=[rmin, vmax], title_text="Speed / m/s", row=1, col=2)
    fig.update_xaxes(title_text="Time / s", row=1, col=2)
    return fig


def plot_fundamental_diagram(country: str, density: pd.DataFrame, speed: pd.DataFrame) -> go.Figure:
    """Plot FD density vs speed."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=density[::50],
            y=speed[::50],
            marker={"color": "blue"},
            mode="markers",
        ),
    )

    # rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    # rmax = np.max(density) + 0.5
    vmin = np.min(speed) - 0.05
    vmax = np.max(speed) + 0.05
    fig.update_layout(
        title=f"Country: {country}",
        xaxis_title="Density / 1/m/m",
        showlegend=False,
    )
    fig.update_yaxes(range=[vmin, vmax], title_text="Speed / m/s")
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
    )

    return fig


def plot_fundamental_diagram_all(country_data: Dict[str, Any]) -> go.Figure:
    """Plot FD for all countries."""
    fig = go.Figure()
    vmax = -1.0

    colors_const = ["blue", "red", "green", "magenta", "orange"]
    marker_shapes = ["circle", "square", "diamond", "cross", "triangle-up"]

    colors = {}
    for country, color in zip(country_data.keys(), colors_const):
        colors[country] = color

    for i, (country, (density, speed)) in enumerate(country_data.items()):
        fig.add_trace(
            go.Scatter(
                x=density[::50],
                y=speed[::50],
                marker={
                    "color": colors[country],
                    "opacity": 0.5,
                    "symbol": marker_shapes[i % len(marker_shapes)],
                },
                mode="markers",
                name=f"{country}",
                showlegend=True,
            )
        )
        vmax = max(vmax, np.max(speed))
        vmin = min(vmax, np.min(speed))

    vmax += 0.05
    vmin -= 0.05

    # vmax = 2.0
    fig.update_yaxes(range=[vmin, vmax], title_text=r"$v\; / \frac{m}{s}$")
    fig.update_xaxes(
        range=[0, 5],
        title_text=r"$\rho / m^{-2}$",
        scaleanchor="y",
        scaleratio=1,
    )

    return fig


def plot_agent_and_neighbors(
    agent: int,
    frame: int,
    rdata: pd.DataFrame,
    neighbors: npt.NDArray[np.float64],
    neighbors_ids: List[int],
    exterior: Polygon,
    interior: Polygon,
    middle_path: Polygon,
    neighbor_type: List[str],
) -> go.Figure:
    """Plot agent and neighbors at <frame>."""
    agent_data = rdata[(rdata["id"] == agent) & (rdata["frame"] == frame)]

    X0 = neighbors[:, 0]
    Y0 = neighbors[:, 1]
    color = {"prev": "blue", "next": "green"}
    dists = []
    dists2 = []
    agent_fake = np.array([])
    X_fake = []
    Y_fake = []

    text = ""
    show_fake = st.checkbox("Show projections", value=True)
    path_distances = hp.precompute_path_distances(middle_path)
    if not agent_data.empty:
        x_agent = agent_data.iloc[0]["x"]
        y_agent = agent_data.iloc[0]["y"]

        for i, (x, y, ni, nt) in enumerate(zip(X0, Y0, neighbors_ids, neighbor_type)):
            dists.append(np.linalg.norm(np.array([x_agent, y_agent]) - np.array([x, y])))
            d, p1, p2 = hp.sum_distances_between_agents_on_path(
                np.array([x_agent, y_agent]),
                np.array([x, y]),
                middle_path,
                path_distances,
            )
            agent_fake = p1
            X_fake.append(p2[0])
            Y_fake.append(p2[1])
            dists2.append(d)
            text += f"<b>{neighbor_type[i]}: </b> [Euklidean {dists[i]:.2} m. Arc {dists2[i]:.2f} m]. "

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[
            # f"<b>Agent {agent} at Frame {frame} has {len(neighbors)} neighbors: {neighbors_ids}</b>"
            text
        ],
        x_title="X",
        y_title=r"Y",
    )
    x_exterior, y_exterior = Polygon(exterior).exterior.xy
    x_exterior = list(x_exterior)
    y_exterior = list(y_exterior)
    x_interior, y_interior = Polygon(interior).exterior.xy
    x_interior = list(x_interior)
    y_interior = list(y_interior)
    x_middle, y_middle = Polygon(middle_path).exterior.xy
    x_middle = list(x_middle)
    y_middle = list(y_middle)

    xmin = np.min(x_exterior)
    xmax = np.max(x_exterior)
    ymin = np.min(y_exterior) - 0.5
    ymax = np.max(y_exterior) + 0.5
    fig.add_trace(
        go.Scatter(
            x=x_exterior,
            y=y_exterior,
            mode="lines",
            line={"color": "red"},
            name="exterior",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_interior,
            y=y_interior,
            mode="lines",
            line={"color": "red"},
            name="interior",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_middle,
            y=y_middle,
            mode="lines",
            line={"color": "black"},
            name="middle",
            showlegend=False,
        )
    )

    data0 = rdata[rdata["frame"] == frame]
    agent_data = rdata[(rdata["id"] == agent) & (rdata["frame"] == frame)]

    X = data0["x"]
    Y = data0["y"]
    idgray = data0["id"]

    for i, x, y in zip(idgray, X, Y):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                marker={"size": 20},
                name=f"ID: {i:0.0f}",
                line_color="black",
                showlegend=False,
            )
        )
    if len(neighbors) > 2:
        hull = spatial.ConvexHull(neighbors)
        X00 = neighbors[hull.vertices, 0]
        Y00 = neighbors[hull.vertices, 1]
    else:
        X00 = X0
        Y00 = Y0

    polygon = go.Scatter(
        x=X00,
        y=Y00,
        showlegend=False,
        mode="lines",
        fill="toself",
        name=f"ConvexHull for pedestrian {agent}",
        line={"color": "LightSeaGreen", "width": 2},
    )

    fig.add_trace(polygon, row=1, col=1)
    # plot neighbors
    for x, y, ni, nt in zip(X0, Y0, neighbors_ids, neighbor_type):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                name=f"{nt}: {ni:0.0f}",
                marker={"size": 20},
                line_color=color[nt],
                showlegend=True,
            )
        )
    if show_fake:
        for x, y, ni, nt in zip(X_fake, Y_fake, neighbors_ids, neighbor_type):
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    name=f"{nt}: {ni:0.0f}",
                    marker={
                        "size": 20,
                        "color": "rgba(255,255,255,0)",
                        "line": {"color": color[nt], "width": 2},
                    },
                    mode="markers+lines",
                    showlegend=False,
                )
            )

    # plot agent
    if not agent_data.empty:
        fig.add_trace(
            go.Scatter(
                x=[x_agent],
                y=[y_agent],
                fillcolor="red",
                name=f"Agent: {agent:0.0f}",
                marker={"size": 20},
                line_color="firebrick",
                showlegend=True,
            )
        )
        if show_fake and agent_fake:
            fig.add_trace(
                go.Scatter(
                    x=[agent_fake[0]],
                    y=[agent_fake[1]],
                    # fillcolor="blue",
                    name=f"Agent: {agent:0.0f}",
                    marker={
                        "size": 20,
                        "color": "rgba(255,255,255,0)",
                        "line": {"color": "firebrick", "width": 2},
                    },
                    # line_color="blue",
                    showlegend=False,
                    mode="markers+lines",
                )
            )

    eps = 0.2
    fig.update_yaxes(
        range=[ymin - eps, ymax + eps],
    )
    fig.update_xaxes(
        range=[xmin - eps, xmax + eps],
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def plot_x_y_trace(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    name: str,
    line_property: str,
) -> go.Figure:
    """Plot two arrays."""
    x = np.unique(x)
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"width": 3, "color": color, "dash": line_property},
        fill="none",
        showlegend=True,
        name=name,
    )
