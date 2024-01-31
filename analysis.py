import pandas as pd
import numpy as np
from pandas import DataFrame
from shapely.geometry import Point
from shapely.ops import unary_union

# from memory_profiler import profile
import streamlit as st


def calculate_speed(data: DataFrame) -> DataFrame:
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
    data["delta_x"] = data.groupby("id")["x"].diff()
    data["delta_y"] = data.groupby("id")["y"].diff()
    data["delta_t"] = data.groupby("id")["time"].diff()

    # Calculate the distance traveled between frames
    data["distance"] = np.sqrt(data["delta_x"] ** 2 + data["delta_y"] ** 2)

    # Calculate speed (distance/time)
    data["speed"] = data["distance"] / data["delta_t"]

    # Handle any NaN values that might arise (e.g., the first frame for each ID)
    data["speed"] = data["speed"].fillna(0)

    return data


def calculate_circular_distance_and_gender(
    data: DataFrame, total_agents: int
) -> DataFrame:
    """
    Calculate the distance to the nearest neighbors considering IDs as circular,
    and also include the gender of these neighbors.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the columns 'ID', 'gender', 'frame', 'x', and 'y'.
    total_agents (int): Total number of agents.

    Returns:
    DataFrame: The input DataFrame with additional columns for distances and gender of neighbors.
    """
    # Sort the data by frame and then by ID
    data = data.sort_values(by=["frame", "id"])

    # Initialize columns for distances to neighbors and their genders
    data["distance_to_prev_neighbor"] = np.nan
    data["gender_of_prev_neighbor"] = None
    data["distance_to_next_neighbor"] = np.nan
    data["gender_of_next_neighbor"] = None

    # Group by frame and calculate distances and genders for each group
    for frame, group in data.groupby("frame"):
        ids = group["id"].to_numpy()
        positions = group[["x", "y"]].to_numpy()
        genders = group["gender"].to_numpy()

        for i, (agent_id, position, gender) in enumerate(zip(ids, positions, genders)):
            # Find the ID of the previous and next neighbors (circular)
            prev_index = i - 1 if i > 0 else -1
            next_index = i + 1 if i < len(ids) - 1 else 0

            # Calculate distances to neighbors
            data.at[group.index[i], "distance_to_prev_neighbor"] = np.linalg.norm(
                position - positions[prev_index]
            )
            data.at[group.index[i], "distance_to_next_neighbor"] = np.linalg.norm(
                position - positions[next_index]
            )

            # Record the genders of the neighbors
            data.at[group.index[i], "gender_of_prev_neighbor"] = genders[prev_index]
            data.at[group.index[i], "gender_of_next_neighbor"] = genders[next_index]

    return data


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
    return union_of_circles.area


def calculate_instantaneous_density_per_frame(data: DataFrame) -> DataFrame:
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
    fps = st.slider("fps", 1, 100, 25, 5)
    for frame, frame_data in data.groupby("frame"):
        if frame % fps != 0:
            continue

        total_union_area = calculate_union_area_shapely(frame_data[["x", "y"]])
        num_pedestrians = frame_data["id"].nunique()
        #        print(num_pedestrians)
        #        print(total_union_area)
        #        print("---")
        instantaneous_density = (
            num_pedestrians / total_union_area if total_union_area else 0
        )
        density_results.append(
            {"frame": frame, "instantaneous_density": instantaneous_density}
        )

    return pd.DataFrame(density_results)
