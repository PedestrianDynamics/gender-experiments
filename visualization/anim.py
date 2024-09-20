# Copyright © 2012-2023 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later

"""This code is used in examples on jupedsim.org.

We make no promises about the functions from this file w.r.t. API stability. We
reservere us the right to change the code here w.o. warning. Do not use the
code here. Use it at your own peril.
"""


from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pedpy
import plotly.graph_objects as go
from plotly.graph_objs import Figure, Scatter
from plotly.graph_objs.layout import Shape
from shapely import Polygon

DUMMY_SPEED = -1000


def _speed_to_color(speed: float, min_speed: float, max_speed: float) -> str:
    """Map a speed value to a color using a colormap."""
    normalized_speed = (speed - min_speed) / (max_speed - min_speed)
    r, g, b = plt.cm.jet_r(normalized_speed)[:3]  # type: ignore
    return f"rgba({r*255:.0f}, {g*255:.0f}, {b*255:.0f}, 0.5)"


def _get_line_color(disk_color: str) -> str:
    """Change line color based on brightness."""
    r, g, b, _ = [int(float(val)) for val in disk_color[5:-2].split(",")]
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "black" if brightness > 127 else "white"


def _create_orientation_line(
    row: pd.DataFrame, line_length: float = 0.2, color: str = "black"
) -> Shape:
    """Create orientation Shape object."""
    end_x = row["x"] + line_length * 0
    end_y = row["y"] + line_length * 0

    return go.layout.Shape(
        type="line",
        x0=row["x"],
        y0=row["y"],
        x1=end_x,
        y1=end_y,
        line={"color": color, "width": 3},
    )


def _get_geometry_traces(area: Polygon) -> Scatter:
    """Construct geometry traces."""
    geometry_traces = []
    x, y = area.exterior.xy
    geometry_traces.append(
        go.Scatter(
            x=np.array(x),
            y=np.array(y),
            mode="lines",
            line={"color": "grey"},
            showlegend=False,
            name="Exterior",
            hoverinfo="name",
        )
    )
    for inner in area.interiors:
        xi, yi = zip(*inner.coords[:])
        geometry_traces.append(
            go.Scatter(
                x=np.array(xi),
                y=np.array(yi),
                mode="lines",
                line={"color": "grey"},
                showlegend=False,
                name="Obstacle",
                hoverinfo="name",
            )
        )
    return geometry_traces


def _get_colormap(
    frame_data: pd.DataFrame, max_speed: float, color_mode: str
) -> List[Scatter]:
    """Utilize scatter plots with varying colors for each agent instead of individual shapes.

    This trace is only to incorporate a colorbar in the plot.
    """
    if color_mode == "Speed":
        marker_dict = {
            "size": frame_data["radius"] * 2,
            "color": frame_data["speed"],
            "colorscale": "Jet_r",
            "colorbar": {"title": "Speed [m/s]"},
            "cmin": 0,
            "cmax": max_speed,
        }
    else:
        colors = frame_data["gender"].map({2: "blue", 1: "green"})
        marker_dict = {
            "size": frame_data["radius"] * 2,
            "color": colors,
            # "colorbar": {"title": "Gender"},
        }

    scatter_trace = go.Scatter(
        x=frame_data["x"],
        y=frame_data["y"],
        mode="markers",
        marker=marker_dict,
        text=frame_data["speed"] if color_mode == "speed" else frame_data["gender"],
        showlegend=False,
        hoverinfo="none",
    )

    return [scatter_trace]


def _get_shapes_for_frame(
    frame_data: pd.DataFrame, min_speed: float, max_speed: float, color_mode: str
) -> Tuple[Shape, Scatter, Shape]:
    """Construct circles as Shapes for agents, Hover and Directions."""

    def create_shape(row: pd.DataFrame) -> Shape:
        """Construct circles as Shapes for agents."""
        hover_trace = go.Scatter(
            x=[row["x"]],
            y=[row["y"]],
            text=[f"ID: {row['id']}, Pos({row['x']:.2f},{row['y']:.2f})"],
            mode="markers",
            marker={"size": 1, "opacity": 1},
            hoverinfo="text",
            showlegend=False,
        )
        if row["speed"] == DUMMY_SPEED:
            dummy_trace = go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers",
                marker={"size": 1, "opacity": 0},
                hoverinfo="none",
                showlegend=False,
            )
            return (
                go.layout.Shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=row["x"] - row["radius"],
                    y0=row["y"] - row["radius"],
                    x1=row["x"] + row["radius"],
                    y1=row["y"] + row["radius"],
                    line={"width": 0},
                    fillcolor="rgba(255,255,255,0)",  # Transparent fill
                ),
                dummy_trace,
                _create_orientation_line(row, color="rgba(255,255,255,0)"),
            )
        if color_mode == "Speed":
            color = _speed_to_color(row["speed"], min_speed, max_speed)
        else:
            gender_colors = {
                1: "blue",  # Assuming 1 is for female
                2: "green",  # Assuming 2 is for male
                0: "black",  # non binary
                -1: "yellow",
            }
            color = gender_colors[row["gender"]]
        return (
            go.layout.Shape(
                type="circle",
                xref="x",
                yref="y",
                x0=row["x"] - row["radius"],
                y0=row["y"] - row["radius"],
                x1=row["x"] + row["radius"],
                y1=row["y"] + row["radius"],
                line_color=color,
                fillcolor=color,
            ),
            hover_trace,
            _create_orientation_line(row, color=color),
        )

    results = frame_data.apply(create_shape, axis=1).tolist()
    # results = frame_data.apply(lambda row: create_shape(color_mode=color_mode), axis=1).tolist()
    shapes = [res[0] for res in results]
    hover_traces = [res[1] for res in results]
    arrows = [res[2] for res in results]
    return shapes, hover_traces, arrows


def _create_fig(
    initial_agent_count: int,
    initial_shapes: Shape,
    initial_arrows: Shape,
    initial_hover_trace: Shape,
    initial_scatter_trace: Shape,
    geometry_traces: Shape,
    frames: pd.DataFrame,
    steps: List[Dict[str, Any]],
    area_bounds: Tuple[float, float, float, float],
    width: int = 800,
    height: int = 800,
    title_note: str = "",
) -> Figure:
    """Creates a Plotly figure with animation capabilities.

    Returns:
        go.Figure: A Plotly figure with animation capabilities.
    """

    minx, miny, maxx, maxy = area_bounds
    title = f"<b>{title_note + '  |  ' if title_note else ''}Number of Agents: {initial_agent_count}</b>"
    fig = go.Figure(
        data=geometry_traces + initial_scatter_trace + initial_hover_trace,
        frames=frames,
        layout=go.Layout(
            shapes=initial_shapes + initial_arrows, title=title, title_x=0.5
        ),
    )
    fig.update_layout(
        updatemenus=[_get_animation_controls()],
        sliders=[_get_slider_controls(steps)],
        autosize=False,
        width=width,
        height=height,
        xaxis={"range": [minx - 0.5, maxx + 0.5]},
        yaxis={"scaleanchor": "x", "scaleratio": 1, "range": [miny - 0.5, maxy + 0.5]},
    )

    return fig


def _get_animation_controls() -> Dict[str, Any]:
    """Returns the animation control buttons for the figure."""
    return {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                    },
                ],
                "label": "Play",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }


def _get_slider_controls(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns the slider controls for the figure."""
    return {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 100, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": steps,
    }


def _get_processed_frame_data(
    data_df: pd.DataFrame, frame_num: int, max_agents: int
) -> Tuple[pd.DataFrame, int]:
    """Process frame data and ensure it matches the maximum agent count."""
    frame_data = data_df[data_df["frame"] == frame_num]
    agent_count = len(frame_data)
    dummy_agent_data = {"x": 0, "y": 0, "radius": 0, "speed": DUMMY_SPEED}
    while len(frame_data) < max_agents:
        dummy_df = pd.DataFrame([dummy_agent_data])
        frame_data = pd.concat([frame_data, dummy_df], ignore_index=True)
    return frame_data, agent_count


def animate(
    data_df: pd.DataFrame,
    area: pedpy.WalkableArea,
    color_mode: str,
    *,
    width: int = 800,
    height: int = 800,
    radius: float = 0.1,
    every_nth_frame: int = 50,
    title_note: str = "",
) -> Figure:
    """Animate a trajectory."""
    data_df["radius"] = radius
    frames = data_df["frame"].unique()
    min_speed = data_df["speed"].min()
    max_speed = data_df["speed"].max()
    max_agents = data_df.groupby("frame").size().max()
    frames = []
    steps = []
    unique_frames = data_df["frame"].unique()
    selected_frames = unique_frames[::every_nth_frame]
    geometry_traces = _get_geometry_traces(area.polygon)
    initial_frame_data = data_df[data_df["frame"] == data_df["frame"].min()]
    initial_agent_count = len(initial_frame_data)
    (
        initial_shapes,
        initial_hover_trace,
        initial_arrows,
    ) = _get_shapes_for_frame(initial_frame_data, min_speed, max_speed, color_mode)
    color_map_trace = _get_colormap(initial_frame_data, max_speed, color_mode)
    for frame_num in selected_frames:
        frame_data, agent_count = _get_processed_frame_data(
            data_df, frame_num, max_agents
        )
        shapes, hover_traces, arrows = _get_shapes_for_frame(
            frame_data, min_speed, max_speed, color_mode
        )
        # title = f"<b>{title_note + '  |  ' if title_note else ''}N: {agent_count}</b>"
        title = f"<b>{title_note + '  |  ' if title_note else ''}Number of Agents: {initial_agent_count}</b>"
        frame_name = str(int(frame_num))
        frame = go.Frame(
            data=geometry_traces + hover_traces,
            name=frame_name,
            layout=go.Layout(
                shapes=shapes + arrows,
                title=title,
                title_x=0.5,
            ),
        )
        frames.append(frame)

        step = {
            "args": [
                [frame_name],
                {
                    "frame": {"duration": 100, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 500},
                },
            ],
            "label": frame_name,
            "method": "animate",
        }
        steps.append(step)

    return _create_fig(
        initial_agent_count,
        initial_shapes,
        initial_arrows,
        initial_hover_trace,
        color_map_trace,
        geometry_traces,
        frames,
        steps,
        area.bounds,
        width=width,
        height=height,
        title_note=title_note,
    )
