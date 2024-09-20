"""Measurements of density, speed."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pedpy as pp
import streamlit as st
from pandas import DataFrame
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import src.utils.helper as hp
import src.visualization.plots as pl


def calculate_speed(data: DataFrame, dv: int) -> DataFrame:
    """
    Calculate the speed of each individual in the dataset.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the columns 'id', 'time', 'x', and 'y'.

    Returns:
    DataFrame: The input DataFrame with additional columns for delta_x, delta_y, delta_t, distance, and speed.
    """
    # Sort the data by ID and then by frame (assuming 'frame' is equivalent to 'time')
    data = data.sort_values(by=["id", "time"])
    # Calculate the difference in position and time for each row
    data["delta_x"] = data.groupby("id")["x"].diff(dv)
    data["delta_y"] = data.groupby("id")["y"].diff(dv)
    data["delta_t"] = data.groupby("id")["time"].diff(dv)

    # Calculate the distance traveled between frames
    data["distance"] = np.sqrt(data["delta_x"] ** 2 + data["delta_y"] ** 2)

    # Calculate speed (distance/time)
    data["speed"] = data["distance"] / data["delta_t"]

    # Handle any NaN values that might arise (e.g., the first frame for each ID)
    # data["speed"] = data["speed"].fillna(0)

    return data


def calculate_individual_density_csv(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Voronoi density."""
    # Replace None values with NaN
    columns = [
        "same_gender_proximity_prev",
        "same_gender_proximity_next",
        "diff_gender_proximity_prev",
        "diff_gender_proximity_next",
    ]
    data.loc[:, columns].replace({None: np.nan}, inplace=True)

    # Calculate half sum distances
    half_sum_distances = 0.5 * (
        data["same_gender_proximity_prev"].fillna(0)
        + data["diff_gender_proximity_prev"].fillna(0)
        + data["same_gender_proximity_next"].fillna(0)
        + data["diff_gender_proximity_next"].fillna(0)
    )

    # Avoid division by zero
    half_sum_distances.replace(0, np.nan, inplace=True)

    # Calculate individual density
    individual_density = 1 / half_sum_distances

    # Create a dataframe with frame and individual density

    return pd.DataFrame(
        {
            "frame": data["frame"],
            "id": data["id"],
            "individual_density": individual_density,
        }
    )


def calculate_union_area_shapely(data: DataFrame, R: float = 0.75) -> float:
    """
    Calculate the total area of the union of circles representing personal spaces using Shapely.

    Parameters:
    data (DataFrame): DataFrame with columns 'x', and 'y'.
    R (float): Radius of personal space in meters.

    Returns:
    float: Area of the union of circles.
    """
    # Create circles for each pedestrian
    circles = [Point(row["x"], row["y"]).buffer(R) for index, row in data.iterrows()]
    #    print(circles)
    # Calculate the union of all circles
    union_of_circles = unary_union(circles)
    # print(union_of_circles)
    # print("----")
    # Calculate and return the area of the union
    return float(union_of_circles.area)


def calculate_instantaneous_density_per_frame(data: DataFrame, fps: int) -> DataFrame:
    """
    Calculate the instantaneous density per frame based on the personal space of each pedestrian.

    Eq.8 in Pouw2024
    High-statistics pedestrian dynamics on stairways and their probabilistic fundamental diagrams

    Parameters:
    data (DataFrame): DataFrame with columns 'id', 'frame', 'x', and 'y'.

    Returns:
    DataFrame: DataFrame with an additional column 'instantaneous_density' for each frame.
    """
    density_results = []
    for frame, frame_data in data.groupby("frame"):
        if frame % fps != 0:
            continue

        total_union_area = calculate_union_area_shapely(frame_data[["x", "y"]])
        num_pedestrians = frame_data["id"].nunique()
        #        print(num_pedestrians)
        #        print(total_union_area)
        #        print("---")
        instantaneous_density = num_pedestrians / total_union_area if total_union_area else 0
        density_results.append({"frame": frame, "instantaneous_density": instantaneous_density})

    return pd.DataFrame(density_results)


def calculate_steady_state(data: pd.DataFrame, window_size: float, threshold: float, diff_const: float) -> int:
    """Calculate the rate of change (first derivative)."""
    # data = data.fillna(method="ffill") # depcretated
    data.ffill()
    rate_of_change = data.diff(diff_const)

    # Calculate the rolling variance or average change
    rolling_variance = rate_of_change.rolling(window=window_size).var()

    # Find where the variance falls below the threshold
    return int(rolling_variance[rolling_variance < threshold].first_valid_index())


def density_speed_time_series_micro(country: str, filename: str, dv: int, diff_const: int) -> None:
    """Calculate the individual density (Voronoi 1D) and show it."""
    msg = st.empty()
    trajectory_data = hp.load_file(filename)
    eps = 0.5
    xmin = trajectory_data.data["x"].min() - eps
    xmax = trajectory_data.data["x"].max() + eps
    ymin = trajectory_data.data["y"].min() - eps
    ymax = trajectory_data.data["y"].max() + eps
    measurement_area = pp.MeasurementArea(Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]))
    result_csv = st.session_state.config.proximity_results_euc["path"]

    if not result_csv.exists():
        msg.warning(f"{result_csv} does not exist yet!")
        with st.status(f"Downloading {result_csv}...", expanded=True):
            hp.download_csv(
                st.session_state.config.proximity_results_euc["url"],
                st.session_state.config.proximity_results_euc["path"],
            )

    if result_csv.exists():
        msg.info(f"Reading file {result_csv}")
        df = pd.read_csv(result_csv)
    #    cs, cd = st.columns((1, 1))
    with st.spinner(f"Calculating {country} ..."):
        start_time = time.time()
        new_path = "/".join(Path(filename).parts[1:])
        filter_df = df[(df["country"] == country) & (df["file"] == new_path)]
        # st.info(f"{file}")
        # st.dataframe(filter_df)
        density = calculate_individual_density_csv(filter_df)
        #       cd.dataframe(density)
        msg.info("calculate speed")
        # speed = calculate_speed(data, dv)
        speed = pp.compute_individual_speed(
            traj_data=trajectory_data,
            frame_step=dv,
            speed_calculation=pp.SpeedCalculation.BORDER_SINGLE_SIDED,
        )
        mean_speed = pp.compute_mean_speed_per_frame(
            traj_data=trajectory_data,
            individual_speed=speed,
            measurement_area=measurement_area,
        )
        msg.info("calculate speed steady state")
        steady_state_index = calculate_steady_state(mean_speed, window_size=5, threshold=0.1, diff_const=diff_const)
        mean_speed = mean_speed.iloc[steady_state_index:-steady_state_index]

        end_time = time.time()
        elapsed_time = end_time - start_time
        msg.info(f"Time taken to calculate speed and density micro: {elapsed_time:.2f} seconds")
        fig = pl.plot_time_series(
            density,
            mean_speed,
            trajectory_data.frame_rate,
            "individual_density",
        )
        st.plotly_chart(fig)


# def kde():
#         filtered_data = proximity_df[(proximity_df['country'] == country) &
#                                      (proximity_df['category'] == category)]

#         # Perform KDE to estimate the PDF
#         data = filtered_data['value'].dropna()  # Ensure no NaN values
#         kde = gaussian_kde(data)
#         x_range = np.linspace(data.min(), data.max(), 500)
#         pdf = kde(x_range)

#         # Plotting with Plotly
#         fig = go.Figure(data=[go.Scatter(x=x_range, y=pdf, mode='lines')])
#         fig.update_layout(
#             title=f'Probability Density Function of {category} in {country}',
#             xaxis_title='Value',
#             yaxis_title='Density'
#         )
#         fig.show()
