import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st
import helper as hp

from shapely import Polygon
import glob
from scipy import spatial


def plot_trajectories(
    data: pd.DataFrame, framerate: int, uid: int, exterior, interior, plot_parcour
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

    # rotated_data = hp.rotate_trajectories(
    #     data,
    #     st.session_state.center_x,
    #     st.session_state.center_y,
    #     st.session_state.angle_degrees,
    # )
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


def plot_time_series(
    data: pd.DataFrame, speed: pd.DataFrame, fps: int, ss_index: int, key_density
) -> go.Figure:
    density = data[key_density]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Mean density: {np.mean(density):.2f} (+- {np.std(density):.2f}) 1/m/m",
            f"Mean speed: {np.mean(speed['speed']):.2f} (+- {np.std(speed['speed']):.2f}) / m/s",
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
            x=speed["time"].loc[ss_index:],
            y=speed["speed"].loc[ss_index:],
            line={"color": "blue"},
            marker={"color": "blue"},
            mode="lines",
        ),
        row=1,
        col=2,
    )

    rmin = 0  # np.min(data["instantaneous_density"]) - 0.5
    rmax = 5  # np.max(data["instantaneous_density"]) + 0.5
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


def plot_fundamental_diagram_all(country_data) -> go.Figure:
    fig = go.Figure()

    rmax = -1
    vmax = -1

    colors_const = ["blue", "red", "green", "magenta", "black"]
    marker_shapes = ["circle", "square", "diamond", "cross", "x-thin"]  # Example shapes

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
        rmax = max(rmax, np.max(density))
        vmax = max(vmax, np.max(speed))
        vmin = min(vmax, np.min(speed))

    vmax += 0.05
    rmax += 0.05
    vmin -= 0.05

    # vmax = 2.0
    fig.update_yaxes(range=[vmin, vmax], title_text=r"$v\; / \frac{m}{s}$")
    fig.update_xaxes(
        title_text=r"$\rho / m^{-2}$",
        scaleanchor="y",
        scaleratio=1,
    )

    return fig


def plot_rudina_fd(countries, fps=100):
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Density - Speed"),
    )
    rmax = -1
    vmax = -1
    rmin = 100
    vmin = 100

    colors_const = ["blue", "red", "green", "magenta", "black"]
    colors = {}
    for country, color in zip(st.session_state.config.countries, colors_const):
        colors[country] = color

    combined_df = {}
    for country in countries:
        files = glob.glob(f"./rho_speed/{country}/2ColData/*.txt")
        df_list = []
        with st.spinner(f"loading files of {country}"):
            for file_path in files:
                df = pd.read_csv(
                    file_path,
                    sep=r"\s+",
                    comment="#",
                    names=["rho", "velocity"],
                )
                df_list.append(df)

        combined_df[country] = pd.concat(df_list, ignore_index=True)

        fig.add_trace(
            go.Scatter(
                x=combined_df[country]["rho"][::fps],
                y=combined_df[country]["velocity"][::fps],
                marker={"color": colors[country]},
                mode="markers",
                name=f"{country}",
                showlegend=True,
                opacity=0.5,
            ),
            row=1,
            col=1,
        )
        rmax = max(rmax, np.max(combined_df[country]["rho"]))
        vmax = max(vmax, np.max(combined_df[country]["velocity"]))
        rmin = min(rmin, np.min(combined_df[country]["rho"]))
        vmin = min(vmin, np.min(combined_df[country]["velocity"]))
        rmax = 8
    fig.update_yaxes(
        range=[vmin - 0.5, vmax + 0.5], title_text="Speed / m/s", row=1, col=1
    )
    fig.update_xaxes(
        range=[rmin - 0.5, rmax + 0.5], title_text="Density / 1/m/m", row=1, col=1
    )
    return fig


def plot_agent_and_neighbors(
    agent,
    frame,
    rdata,
    neighbors,
    neighbors_ids,
    exterior,
    interior,
    middle_path,
    neighbor_type,
):

    agent_data = rdata[(rdata["id"] == agent) & (rdata["frame"] == frame)]

    X0 = neighbors[:, 0]
    Y0 = neighbors[:, 1]
    color = {"prev": "blue", "next": "green"}
    dists = []
    dists2 = []
    agent_fake = []
    X_fake = []
    Y_fake = []

    text = ""
    show_fake = st.checkbox("Show projections", value=True)
    path_distances = hp.precompute_path_distances(middle_path)
    if not agent_data.empty:
        x_agent = agent_data.iloc[0]["x"]
        y_agent = agent_data.iloc[0]["y"]

        for i, (x, y, ni, nt) in enumerate(zip(X0, Y0, neighbors_ids, neighbor_type)):
            dists.append(
                np.linalg.norm(np.array([x_agent, y_agent]) - np.array([x, y]))
            )
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
    #     fig.add_shape(
    #         type="circle",
    #         xref="x",
    #         yref="y",
    #         x0=x - rped,
    #         y0=y - rped,
    #         x1=x + rped,
    #         y1=y + rped,
    #         fillcolor="Gray",
    #         line_color="lightgray",
    #     )

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

    fig.append_trace(polygon, row=1, col=1)
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
        if show_fake:
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
    # fig.add_shape(
    #     type="circle",
    #     xref="x",
    #     yref="y",
    #     x0=x_agent - rped,
    #     y0=y_agent - rped,
    #     x1=x_agent + rped,
    #     y1=y_agent + rped,
    #     fillcolor="red",
    #     name=f"Agent: {agent:0.0f}",
    #     line_color="firebrick",
    # )

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


def plot_x_y_trace(x, y, title, xlabel, ylabel, color, name, line_property):

    x = np.unique(x)
    # if threshold:
    #     trace_threshold = go.Scatter(
    #         x=[x[0], x[-1]],
    #         y=[threshold, threshold],
    #         mode="lines",
    #         name="Social Distance = 1.5 m",
    #         line=dict(width=4, dash="dash", color="gray"),
    #     )

    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"width": 3, "color": color, "dash": line_property},
        fill="none",
        showlegend=True,
        name=name,
    )


def plot_x_y(x, y, title, xlabel, ylabel, threshold=0):

    x = np.unique(x)
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"<b>{title}</b>"],
        x_title=xlabel,
        y_title=ylabel,
    )
    if threshold:
        trace_threshold = go.Scatter(
            x=[x[0], x[-1]],
            y=[threshold, threshold],
            mode="lines",
            showlegend=True,
            name="Social Distance = 1.5 m",
            line={"width": 4, "dash": "dash", "color": "gray"},
        )
        fig.append_trace(trace_threshold, row=1, col=1)

    trace = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        showlegend=False,
        line={"width": 3, "color": "blue"},
        fill="none",
    )

    fig.append_trace(trace, row=1, col=1)
    return fig
