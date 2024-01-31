import pandas as pd
import numpy as np
from pandas import DataFrame
from shapely.geometry import Point
from shapely.ops import unary_union


def calculate_speed(data: DataFrame) -> DataFrame:
    """
    Calculate the speed of each individual in the dataset.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the columns 'ID', 't(s)', 'x(m)', and 'y(m)'.

    Returns:
    DataFrame: The input DataFrame with additional columns for delta_x, delta_y, delta_t, distance, and speed.
    """
    # Sort the data by ID and then by frame (assuming 'frame' is equivalent to 't(s)')
    data = data.sort_values(by=["ID", "t(s)"])

    # Calculate the difference in position and time for each row
    data["delta_x"] = data.groupby("ID")["x(m)"].diff()
    data["delta_y"] = data.groupby("ID")["y(m)"].diff()
    data["delta_t"] = data.groupby("ID")["t(s)"].diff()

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
    data (DataFrame): A pandas DataFrame containing the columns 'ID', 'gender', 'frame', 'x(m)', and 'y(m)'.
    total_agents (int): Total number of agents.

    Returns:
    DataFrame: The input DataFrame with additional columns for distances and gender of neighbors.
    """
    # Sort the data by frame and then by ID
    data = data.sort_values(by=["frame", "ID"])

    # Initialize columns for distances to neighbors and their genders
    data["distance_to_prev_neighbor"] = np.nan
    data["gender_of_prev_neighbor"] = None
    data["distance_to_next_neighbor"] = np.nan
    data["gender_of_next_neighbor"] = None

    # Group by frame and calculate distances and genders for each group
    for frame, group in data.groupby("frame"):
        ids = group["ID"].to_numpy()
        positions = group[["x(m)", "y(m)"]].to_numpy()
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
    data (DataFrame): DataFrame with columns 'x(m)', and 'y(m)'.
    R (float): Radius of personal space in meters.

    Returns:
    float: Area of the union of circles.
    """
    # Create circles for each pedestrian
    circles = [
        Point(row["x(m)"], row["y(m)"]).buffer(R) for index, row in data.iterrows()
    ]

    # Calculate the union of all circles
    union_of_circles = unary_union(circles)

    # Calculate and return the area of the union
    return union_of_circles.area


def calculate_instantaneous_density_per_frame(data: DataFrame) -> DataFrame:
    """
    Calculate the instantaneous density per frame based on the personal space of each pedestrian.

    Parameters:
    data (DataFrame): DataFrame with columns 'ID', 'frame', 'x(m)', and 'y(m)'.

    Returns:
    DataFrame: DataFrame with an additional column 'instantaneous_density' for each frame.
    """
    density_results = []

    for frame, frame_data in data.groupby("frame"):
        total_union_area = calculate_union_area_shapely(frame_data[["x(m)", "y(m)"]])
        num_pedestrians = frame_data["ID"].nunique()
        instantaneous_density = (
            num_pedestrians / total_union_area if total_union_area else 0
        )
        density_results.append(
            {"frame": frame, "instantaneous_density": instantaneous_density}
        )

    return pd.DataFrame(density_results)
