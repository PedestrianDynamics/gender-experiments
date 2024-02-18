import glob
import itertools
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pedpy
from joblib import Parallel, delayed
from tqdm import tqdm

import helper as hp

exterior, interior, middle_path = hp.generate_parcour()
path_distances = hp.precompute_path_distances(middle_path)


@dataclass
class InitData:
    countries: list
    files: dict
    result_csv: Path
    fps: int


def load_file(file: str) -> pedpy.TrajectoryData:
    """Loads and processes a file to create a TrajectoryData object."""

    data = pd.read_csv(file)
    rename_mapping = {
        "ID": "id",
        "t(s)": "time",
        "x(m)": "x",
        "y(m)": "y",
    }
    column_types = {
        "id": int,
        "gender": int,
        "time": float,
        "x": float,
        "y": float,
    }
    data.rename(columns=rename_mapping, inplace=True)
    hp.set_column_types(data, column_types)

    fps = hp.calculate_fps(data)
    trajectory_data = pedpy.TrajectoryData(data=data, frame_rate=fps)
    return trajectory_data


def filter_frames(data: pd.DataFrame, nagents: int) -> pd.DataFrame:
    """Make sure the data at a frame has the same amount of agents.

    This is necessary, since in some experiments people enter one by one.
    Which means that for instance in frame 1 there is 1 agent,
    in frame 2 2 agents and so on.
    With this method with basically skip all these frames and keep
    only these with <nagents> agents
    """

    # Group data by frame and count the unique IDs in each frame
    frame_counts = data.groupby("frame")["id"].nunique()

    # Filter frames where the count of unique IDs is less than the expected number of agents
    frames_with_all_agents = frame_counts[frame_counts >= nagents].index

    # Filter the original DataFrame based on the selected frames
    cleaned_data = data[data["frame"].isin(frames_with_all_agents)]

    return cleaned_data


def calculate_circular_distance_and_gender_pal(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance to the nearest neighbors based on preprocessed neighbor information,
    considering the spatial arrangement, and include the gender of these neighbors.

    This function is designed for pal experiments, since in these experiments
    agents enter and leave the experiment area, unlike the other experiments.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the columns 'id', 'gender', 'x', and 'y'.

    Returns:
    DataFrame: The input DataFrame with additional columns for distances to the previous and next neighbors
               and the gender of these neighbors.
    """
    # Ensure DataFrame is sorted by frame
    data = data.sort_values(by="frame")

    # Group data by id
    grouped_data = data.groupby("id")

    # Create new columns
    new_columns = [
        "distance_to_prev_neighbor",
        "gender_of_prev_neighbor",
        "distance_to_next_neighbor",
        "gender_of_next_neighbor",
    ]
    data.loc[:, new_columns] = np.nan
    # Iterate over groups
    for id_, group in grouped_data:
        mask = data["id"] == id_
        prev_id = group["prev"].iloc[0]
        next_id = group["next"].iloc[0]
        if np.isnan(prev_id) or np.isnan(next_id):
            continue

        frames_with_both_neighbors = set(group["frame"]).intersection(
            set(grouped_data.get_group(prev_id)["frame"]),
            set(grouped_data.get_group(next_id)["frame"]),
        )

        for frame in frames_with_both_neighbors:
            frame_data = group[group["frame"] == frame]
            distances_to_prev = np.linalg.norm(
                frame_data[["x", "y"]].values
                - grouped_data.get_group(prev_id)[
                    grouped_data.get_group(prev_id)["frame"] == frame
                ][["x", "y"]].values,
                axis=1,
            )
            gender_of_prev = (
                grouped_data.get_group(prev_id)
                .loc[grouped_data.get_group(prev_id)["frame"] == frame, "gender"]
                .values[0]
            )
            distances_to_next = np.linalg.norm(
                frame_data[["x", "y"]].values
                - grouped_data.get_group(next_id)[
                    grouped_data.get_group(next_id)["frame"] == frame
                ][["x", "y"]].values,
                axis=1,
            )
            gender_of_next = (
                grouped_data.get_group(next_id)
                .loc[grouped_data.get_group(next_id)["frame"] == frame, "gender"]
                .values[0]
            )

            data.loc[mask & (data["frame"] == frame), "distance_to_prev_neighbor"] = (
                distances_to_prev
            )
            data.loc[mask & (data["frame"] == frame), "gender_of_prev_neighbor"] = (
                gender_of_prev
            )
            data.loc[mask & (data["frame"] == frame), "distance_to_next_neighbor"] = (
                distances_to_next
            )
            data.loc[mask & (data["frame"] == frame), "gender_of_next_neighbor"] = (
                gender_of_next
            )

    return data


def calculate_circular_distance_and_gender_vect(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance to the nearest neighbors based on preprocessed neighbor information,
    considering the spatial arrangement, and include the gender of these neighbors.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the columns 'id', 'gender', 'x', and 'y'.
    neighbors (dict): A dictionary mapping each ID to its previous and next neighbors.

    Returns:
    DataFrame: The input DataFrame with additional columns for distances to the previous and next neighbors
               and the gender of these neighbors.
    """
    # Create dictionaries to store distances and genders
    ids = data["id"].unique()
    new_columns = [
        "distance_to_prev_neighbor",
        "gender_of_prev_neighbor",
        "distance_to_next_neighbor",
        "gender_of_next_neighbor",
    ]
    data.loc[:, new_columns] = np.nan
    for id_ in ids:

        mask = data["id"] == id_
        id_data = data.loc[mask]
        prev_id = id_data["prev"].iloc[0]
        next_id = id_data["next"].iloc[0]
        # print(id_, prev_id, next_id)
        if np.isnan(prev_id) or np.isnan(next_id):
            continue
        # Calculate distances and genders for previous neighbors
        prev_data = data.loc[data["id"] == prev_id]
        distances_to_prev = np.linalg.norm(
            id_data[["x", "y"]].values - prev_data[["x", "y"]].values, axis=1
        )
        gender_of_prev = prev_data["gender"].astype(int).values
        # Calculate distances and genders for next neighbors
        next_data = data[data["id"] == next_id]
        distances_to_next = np.linalg.norm(
            id_data[["x", "y"]].values - next_data[["x", "y"]].values, axis=1
        )
        gender_of_next = next_data["gender"].astype(int).values
        # Enhance the data with the new columns
        data.loc[mask, "distance_to_prev_neighbor"] = distances_to_prev
        data.loc[mask, "distance_to_next_neighbor"] = distances_to_next
        data.loc[mask, "gender_of_prev_neighbor"] = gender_of_prev
        data.loc[mask, "gender_of_next_neighbor"] = gender_of_next

    return data


def calculate_circular_distance_and_gender_arc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance to the nearest neighbors based on preprocessed neighbor information,
    considering the spatial arrangement, and include the gender of these neighbors.

    Parameters:
    data (DataFrame): A pandas DataFrame containing the columns 'id', 'gender', 'x', and 'y'.
    neighbors (dict): A dictionary mapping each ID to its previous and next neighbors.

    Returns:
    DataFrame: The input DataFrame with additional columns for distances to the previous and next neighbors
               and the gender of these neighbors.
    """
    # Create dictionaries to store distances and genders
    ids = data["id"].unique()
    new_columns = [
        "distance_to_prev_neighbor",
        "gender_of_prev_neighbor",
        "distance_to_next_neighbor",
        "gender_of_next_neighbor",
    ]
    data.loc[:, new_columns] = np.nan
    for id_ in ids:

        mask = data["id"] == id_
        id_data = data.loc[mask]
        prev_id = id_data["prev"].iloc[0]
        next_id = id_data["next"].iloc[0]
        # print(id_, prev_id, next_id)
        if np.isnan(prev_id) or np.isnan(next_id):
            continue
        # Calculate distances and genders for previous neighbors
        prev_data = data.loc[data["id"] == prev_id]
        distances_to_prev = []
        distances_to_next = []
        for p1, p2 in zip(id_data[["x", "y"]].values, prev_data[["x", "y"]].values):
            distance, _, _ = hp.sum_distances_between_agents_on_path(
                p1, p2, middle_path, path_distances
            )
            distances_to_prev.append(distance)

        gender_of_prev = prev_data["gender"].astype(int).values
        # Calculate distances and genders for next neighbors
        next_data = data[data["id"] == next_id]
        for p1, p2 in zip(id_data[["x", "y"]].values, next_data[["x", "y"]].values):
            distance, _, _ = hp.sum_distances_between_agents_on_path(
                p1, p2, middle_path, path_distances
            )
            distances_to_next.append(distance)

        gender_of_next = next_data["gender"].astype(int).values
        # Enhance the data with the new columns
        data.loc[mask, "distance_to_prev_neighbor"] = distances_to_prev
        data.loc[mask, "distance_to_next_neighbor"] = distances_to_next
        data.loc[mask, "gender_of_prev_neighbor"] = gender_of_prev
        data.loc[mask, "gender_of_next_neighbor"] = gender_of_next

    return data


def extract_first_number(filename: str) -> int:
    """Define a regular expression pattern to match the first sequence of digits."""
    pattern = r"\d+"
    numbers = re.findall(pattern, filename)

    return int(numbers[0])


def calculate_proximity_analysis(
    country: str, filename: str, data: pd.DataFrame, fps: int = 25
) -> List[Dict]:
    """
    Performs proximity analysis on given data of agents.
    Filtering by frames and categorizing
    by gender proximity for each agent within the selected frames.

    Args:
        country (str): The country code or name related to the dataset.
        filename (str): The name of the file containing the dataset, used to infer gender composition.
        data (pd.DataFrame): A pandas DataFrame witqh rotated data including agent positions and genders.
        fps (int, optional): The frames per second rate used to filter the data. Defaults to 25.

    Returns:
        List[Dict]: A list of dictionaries, each containing the proximity analysis results for an agent in
        the filtered frames, including country, file name, agent type, ID, frame number, and distances to
        the next and previous neighbors classified by gender similarity.

    Note:
        The function expects 'data' DataFrame to include 'frame', 'id', 'gender',
        'gender_of_next_neighbor', 'gender_of_prev_neighbor', 'distance_to_next_neighbor',
        and 'distance_to_prev_neighbor' columns.
    """
    if country != "pal":
        nagents = extract_first_number(filename)
        cleaned_data = filter_frames(data, nagents)
        processed_data = calculate_circular_distance_and_gender_arc(cleaned_data)
        fps = 25
    else:
        processed_data = calculate_circular_distance_and_gender_pal(data)
        fps = 1

    proximity_analysis_res = []
    # print(processed_data)
    frames_to_include = set(range(0, processed_data["frame"].max(), fps))

    # Filter the DataFrame to only include the desired frames
    filtered_data = processed_data[processed_data["frame"].isin(frames_to_include)]

    # Now iterate over the filtered DataFrame
    if "female" in filename:
        name = "female"
    elif "male" in filename:
        name = "male"
    elif "mix_sorted" in filename:
        name = "mix_sorted"
    elif "mix_random" in filename:
        name = "mix_random"
    else:
        name = "unknown"

    for i, row in filtered_data.iterrows():
        # for i, row in processed_data.iterrows():
        # Check proximity with the next neighbor
        if row["gender"] == row["gender_of_next_neighbor"]:
            same_gender_proximity_next = row["distance_to_next_neighbor"]
        else:
            same_gender_proximity_next = np.nan

        if row["gender"] != row["gender_of_next_neighbor"]:
            diff_gender_proximity_next = row["distance_to_next_neighbor"]
        else:
            diff_gender_proximity_next = np.nan

        # Check proximity with the previous neighbor
        if row["gender"] == row["gender_of_prev_neighbor"]:
            same_gender_proximity_prev = row["distance_to_prev_neighbor"]
        else:
            same_gender_proximity_prev = np.nan

        if row["gender"] != row["gender_of_prev_neighbor"]:
            diff_gender_proximity_prev = row["distance_to_prev_neighbor"]
        else:
            diff_gender_proximity_prev = np.nan

        proximity_analysis_res.append(
            {
                "country": country,
                "file": filename,
                "type": name,
                "id": row["id"],
                "frame": row["frame"],
                "same_gender_proximity_next": same_gender_proximity_next,
                "diff_gender_proximity_next": diff_gender_proximity_next,
                "same_gender_proximity_prev": same_gender_proximity_prev,
                "diff_gender_proximity_prev": diff_gender_proximity_prev,
            }
        )

    return proximity_analysis_res


def unpack_and_process(args):
    return calculate_proximity_analysis(*args)


def prepare_data(country, selected_file, fps):
    trajectory_data = load_file(selected_file)
    data = trajectory_data.data
    return country, selected_file, data, fps


def calculate_with_joblib(init_data: InitData):
    # Prepare tasks

    tasks = []
    for country in init_data.countries:
        print(f"prepare tasks: {country}")
        for filename in init_data.files[country]:
            tasks.append(prepare_data(country, filename, init_data.fps))

    # Define a function to be executed in parallel
    def process_task(task):
        return unpack_and_process(task)

    nproc = -1
    print(f"Running tasks in parallel {len(tasks)} with {nproc} proc...")
    results = Parallel(n_jobs=nproc)(
        delayed(process_task)(task) for task in tqdm(tasks)
    )
    # for task in tasks:
    #     process_task(task)

    print(f"Done running tasks in parallel {len(tasks)} ...")
    # Return the final results
    flattened_results = list(itertools.chain.from_iterable(results))
    flattened_results = pd.DataFrame(flattened_results)
    print(init_data.result_csv)
    flattened_results.to_pickle(init_data.result_csv)

    return flattened_results


def init() -> InitData:
    """
    Initializes the application by setting up the countries, file paths, result CSV path, and FPS (frames per second).

    Returns:
        InitData: A data class containing initialized data including countries, files dictionary, result CSV path, and FPS.
    """
    result_csv = Path("app_data/proximity_analysis_results.csv")
    fps = 25  # For distance calculations, calculate every fps-frame
    countries = ["aus"]  # , "ger", "jap", "chn", "pal"]
    files = {}
    for country in countries:
        files[country] = [str(path) for path in Path(country).glob("*.csv")]

    return InitData(countries=countries, files=files, result_csv=result_csv, fps=fps)


if __name__ == "__main__":
    init_data = init()
    start_time = time.time()
    proximity_df = calculate_with_joblib(init_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    proximity_df.to_csv(init_data.result_csv, index=False)
