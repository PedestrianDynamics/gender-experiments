import pandas as pd
import numpy as np
from pandas import DataFrame
from shapely.geometry import Point
from shapely.ops import unary_union

import os


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


import numpy as np
import pandas as pd


# Example usage:
# neighbors = preprocess_neighbors(your_dataframe)
# updated_data = calculate_circular_distance_and_gender(your_dataframe, neighbors)


# def calculate_circular_distance_and_gender(data: DataFrame) -> DataFrame:
#     """
#     Calculate the distance to the nearest neighbors based on "prev" and "next" columns,
#     considering the spatial arrangement, and include the gender of these neighbors.

#     Parameters:
#     data (DataFrame): A pandas DataFrame containing the columns 'id', 'gender', 'frame', 'x', 'y', 'prev', 'next'.

#     Returns:
#     DataFrame: The input DataFrame with additional columns for distances to the previous and next neighbors
#                and the gender of these neighbors.
#     """

#     # Initialize columns for distances to neighbors and their genders
#     data["distance_to_prev_neighbor"] = np.nan
#     data["gender_of_prev_neighbor"] = None
#     data["distance_to_next_neighbor"] = np.nan
#     data["gender_of_next_neighbor"] = None

#     #    data["prev"] = data["prev"].astype(int)
#     #    data["next"] = data["next"].astype(int)

#     # Calculate distances and gender information based on "prev" and "next" neighbors
#     for index, row in data.iterrows():
#         # print(f"{index}/{len(data)}")
#         # Handle the case where there are no previous or next neighbors
#         if row["prev"] != -1:
#             prev_row = data.loc[data["id"] == row["prev"]].iloc[0]
#             data.at[index, "distance_to_prev_neighbor"] = np.linalg.norm(
#                 [row["x"] - prev_row["x"], row["y"] - prev_row["y"]]
#             )
#             data.at[index, "gender_of_prev_neighbor"] = prev_row["gender"]

#         if row["next"] != -1:
#             next_row = data.loc[data["id"] == row["next"]].iloc[0]
#             data.at[index, "distance_to_next_neighbor"] = np.linalg.norm(
#                 [row["x"] - next_row["x"], row["y"] - next_row["y"]]
#             )
#             data.at[index, "gender_of_next_neighbor"] = next_row["gender"]

#     return data


def calculate_individual_density_csv(data):
    # Replace None values with NaN
    columns = [
        "same_gender_proximity_prev",
        "same_gender_proximity_next",
        "diff_gender_proximity_prev",
        "diff_gender_proximity_next",
    ]
    # data.loc[:, "same_gender_proximity_prev"].replace({None: np.nan}, inplace=True)
    # data.loc[:, "same_gender_proximity_next"].replace({None: np.nan}, inplace=True)
    # data.loc[:, "diff_gender_proximity_prev"].replace({None: np.nan}, inplace=True)
    # data.loc[:, "diff_gender_proximity_next"].replace({None: np.nan}, inplace=True)
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

    frame_density_data = pd.DataFrame(
        {
            "frame": data["frame"],
            "id": data["id"],
            "individual_density": individual_density,
        }
    )

    return frame_density_data


def enhance_data(data):
    # Calculate speed
    data = calculate_speed(data)

    # Calculate density
    data = calculate_individual_density_csv(data)

    return data


# def calculate_individual_density(data: DataFrame) -> DataFrame:
#     """
#     Calculates the individual density for each entity per frame in the dataset. The individual density
#     is defined as the inverse of the sum of half the distance to the previous neighbor and half the distance
#     to the next neighbor.

#     Parameters:
#     - data (DataFrame): A pandas DataFrame containing the columns 'frame', 'x', 'y', 'prev', 'next',
#                         along with 'distance_to_prev_neighbor' and 'distance_to_next_neighbor' if not already calculated.
#     - fps (int): The frames per second of the dataset, used for time-based calculations if necessary.

#     Returns:
#     - DataFrame: The input DataFrame with an additional column 'individual_density' representing the
#                  calculated individual density for each row/entity.
#     """

#     data = calculate_circular_distance_and_gender(data)  #

#     # Calculate half the sum of distances to previous and next neighbors
#     half_sum_distances = 0.5 * (
#         data["distance_to_prev_neighbor"] + data["distance_to_next_neighbor"]
#     )

#     # Avoid division by zero
#     half_sum_distances.replace(0, np.nan, inplace=True)

#     # Calculate individual density
#     data["individual_density"] = 1 / half_sum_distances
#     frame_density_data = data[["frame", "individual_density"]]

#     return frame_density_data


def load_or_calculate_individual_density(
    data: pd.DataFrame, fps: int, filepath: str
) -> pd.DataFrame:
    if os.path.exists(filepath):
        # Load the DataFrame if it exists
        return pd.read_pickle(filepath)
    else:
        # Calculate the density
        data = calculate_individual_density(data)
        # Save the DataFrame for future use
        data.to_pickle(filepath)
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
        instantaneous_density = (
            num_pedestrians / total_union_area if total_union_area else 0
        )
        density_results.append(
            {"frame": frame, "instantaneous_density": instantaneous_density}
        )

    return pd.DataFrame(density_results)


def calculate_steady_state(data, window_size, threshold, diff_const):
    """Calculate the rate of change (first derivative)"""

    # data = data.fillna(method="ffill") # depcretated
    data.ffill()
    rate_of_change = data.diff(diff_const)

    # Calculate the rolling variance or average change
    rolling_variance = rate_of_change.rolling(window=window_size).var()

    # Find where the variance falls below the threshold
    steady_state_index = rolling_variance[
        rolling_variance < threshold
    ].first_valid_index()

    return steady_state_index


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
