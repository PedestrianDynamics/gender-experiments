import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
import helper as hp

from shapely import Polygon


def plot_trajectories(
    data: pd.DataFrame, framerate: int, uid: int, exterior, interior
) -> go.Figure:
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

    rotated = c1.checkbox("Rotated", value=True)
    plot_parcour = c2.checkbox("Parcour", value=True)
    do_animate = c3.checkbox("Animate", value=False)
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

    rotated_data = hp.rotate_trajectories(
        data,
        st.session_state.center_x,
        st.session_state.center_y,
        st.session_state.angle_degrees,
    )
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
                    line=dict(color=color_choice),
                    marker=dict(color=color_choice),
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
                    line=dict(color=color_choice),
                    marker=dict(color=color_choice),
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
                        line=dict(color=color_choice),
                        marker=dict(color=color_choice),
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
                        line=dict(color=color_choice),
                        marker=dict(color=color_choice),
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
                line=dict(color="red"),
                name="exterior",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_interior,
                y=y_interior,
                mode="lines",
                line=dict(color="red"),
                name="interior",
            )
        )
    xmin = np.min(x_exterior)
    xmax = np.max(x_exterior)
    ymin = np.min(y_exterior) - 0.5
    ymax = np.max(y_exterior) + 0.5

    fig.update_layout(
        title=f" Trajectories: {num_agents} | M: {num_unique_males} | F: {num_unique_females} | N: {num_unique_unknowns}",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", range=[xmin, xmax]),
        yaxis=dict(scaleratio=1, range=[ymin, ymax]),
        showlegend=False,
    )
    fig.add_annotation(
        x=0.5,
        y=1.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        text="<span style='color:green;'>Green: Male</span>, <span style='color:blue;'>Blue: Female</span>",
        font=dict(size=12),
        align="center",
    )
    return fig, do_animate


def plot_time_series(
    data: pd.DataFrame, speed: pd.DataFrame, fps: int, ss_index: int
) -> go.Figure:
    density = data["instantaneous_density"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Mean density: {np.mean(density):.2f} (+- {np.std(density):.2f}) 1/m/m",
            f"Mean speed: {np.mean(speed['speed']):.2f} (+- {np.std(speed['speed']):.2f}) / m/s",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data["frame"] / fps,
            y=density,
            line=dict(color="blue"),
            marker=dict(color="blue"),
            mode="lines",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=speed["time"].loc[ss_index:],
            y=speed["speed"].loc[ss_index:],
            line=dict(color="blue"),
            marker=dict(color="blue"),
            mode="lines",
        ),
        row=1,
        col=2,
    )

    rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    rmax = 3  # np.max(data["instantaneous_density"]) + 0.5
    vmax = np.max(speed["speed"]) + 0.5
    fig.update_layout(
        xaxis_title="Time / s",
        showlegend=False,
    )
    fig.update_yaxes(range=[rmin, rmax], title_text="Density / 1/m/m", row=1, col=1)
    fig.update_yaxes(range=[rmin, vmax], title_text="Speed / m/s", row=1, col=2)
    fig.update_xaxes(title_text="Time / s", row=1, col=2)
    return fig


def plot_fundamental_diagram(
    country, density: pd.DataFrame, speed: pd.DataFrame
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Density - Speed", "Density - Speed*Density"),
    )

    fig.add_trace(
        go.Scatter(
            x=density,
            y=speed,
            marker=dict(color="blue"),
            mode="markers",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=density,
            y=speed * density,
            marker=dict(color="blue"),
            mode="markers",
        ),
        row=1,
        col=2,
    )

    # rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    rmax = np.max(density) + 0.5
    vmin = 0
    vmax = np.max(speed) + 0.5
    fig.update_layout(
        title=f"Country: {country}",
        xaxis_title="Density / 1/m/m",
        showlegend=False,
    )
    fig.update_yaxes(range=[vmin, vmax], title_text="Speed / m/s", row=1, col=1)
    fig.update_yaxes(title_text="Speed * Density / 1/m/s", row=1, col=2)
    fig.update_xaxes(title_text="Density / 1/m/m", row=1, col=2)
    return fig


def plot_fundamental_diagram_all(country_data) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Density - Speed", "Density - Speed*Density"),
    )
    rmax = -1
    vmax = -1

    colors_const = ["blue", "red", "green", "magenta", "black"]
    colors = {}
    for country, color in zip(country_data.keys(), colors_const):
        colors[country] = color

    for country, (speed, density) in country_data.items():
        fig.add_trace(
            go.Scatter(
                x=density,
                y=speed,
                marker=dict(color=colors[country]),
                mode="markers",
                name=f"{country}",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=density,
                y=speed * density,
                marker=dict(color=colors[country]),
                mode="markers",
                name=f"{country}",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        rmax = max(rmax, np.max(density))
        vmax = max(vmax, np.max(speed))

    vmin = 0
    vmax += 0.5
    rmax += 0.5
    fig.update_layout(
        # title=f"Country: {country}",
        xaxis_title="Density / 1/m/m",
    )
    fig.update_yaxes(range=[vmin, vmax], title_text="Speed / m/s", row=1, col=1)
    fig.update_yaxes(title_text="Speed * Density / 1/m/s", row=1, col=2)
    fig.update_xaxes(range=[0, rmax], title_text="Density / 1/m/m", row=1, col=1)
    fig.update_xaxes(range=[0, rmax], title_text="Density / 1/m/m", row=1, col=2)
    return fig
