"""Standalone script to calculate the distances and output the giant csv file for all countries and all files."""

import itertools
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import pedpy
from joblib import Parallel, delayed
from tqdm import tqdm
from enum import Enum

parent_dir = Path(__file__).resolve().parents[2]
print("Parent dir", parent_dir)
if parent_dir not in sys.path:
    print(parent_dir)
    sys.path.append(f"{parent_dir}/src")



from utils.helper import (
    calculate_fps,
    generate_parcour,
    precompute_path_distances,
    set_column_types,
    sum_distances_between_agents_on_path,
)

_, _, middle_path = generate_parcour()
# print(f"middle_path {len(middle_path)}")
path_distances = precompute_path_distances(middle_path)
# print(f"distances {len(path_distances)}")


class Method(Enum):
    """Method to calculate distances. vect and merge are Euc."""

    VECT = "vect"
    ARC = "arc"
    MERGE = "merge"


@dataclass
class InitData:
    """Class to hold data."""

    countries: List[str]
    files: Dict[str, List[str]]
    result_csv: Path
    fps: int
    method: str



def load_file(file: str) -> pedpy.TrajectoryData:
    """Load and processes a file to create a TrajectoryData object."""
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
    set_column_types(data, column_types)

    fps = calculate_fps(data)
    return pedpy.TrajectoryData(data=data, frame_rate=fps)


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
    return data[data["frame"].isin(frames_with_all_agents)]


def calculate_circular_distance_and_gender_pal(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance to the nearest neighbors based on preprocessed neighbor information.

    This function is special for pal data, since these data are linear and

    This function is designed for pal experiments, since in these experiments
    agents enter and leave the experiment area, unlike the other experiments.
    Agents may not have neighbors (id_neighbor == -1)

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
        # some agents do not have neighbors.
        if np.isnan(prev_id) or np.isnan(next_id):
            continue
        if prev_id == -1 or next_id == -1:
            continue

        frames_with_both_neighbors = set(group["frame"]).intersection(
            set(grouped_data.get_group(prev_id)["frame"]),
            set(grouped_data.get_group(next_id)["frame"]),
        )

        for frame in frames_with_both_neighbors:
            frame_data = group[group["frame"] == frame]
            distances_to_prev = np.linalg.norm(
                frame_data[["x", "y"]].values - grouped_data.get_group(prev_id)[grouped_data.get_group(prev_id)["frame"] == frame][["x", "y"]].values,
                axis=1,
            )
            gender_of_prev = grouped_data.get_group(prev_id).loc[grouped_data.get_group(prev_id)["frame"] == frame, "gender"].values[0]
            distances_to_next = np.linalg.norm(
                frame_data[["x", "y"]].values - grouped_data.get_group(next_id)[grouped_data.get_group(next_id)["frame"] == frame][["x", "y"]].values,
                axis=1,
            )
            gender_of_next = grouped_data.get_group(next_id).loc[grouped_data.get_group(next_id)["frame"] == frame, "gender"].values[0]

            data.loc[mask & (data["frame"] == frame), "distance_to_prev_neighbor"] = distances_to_prev
            data.loc[mask & (data["frame"] == frame), "gender_of_prev_neighbor"] = gender_of_prev
            data.loc[mask & (data["frame"] == frame), "distance_to_next_neighbor"] = distances_to_next
            data.loc[mask & (data["frame"] == frame), "gender_of_next_neighbor"] = gender_of_next

    return data


def compute_distance_merge(data: pd.DataFrame) -> pd.DataFrame:
    """Faster calculation of the distances."""
    all_data = (
        data.merge(
            data[["id", "frame", "gender", "x", "y"]],
            left_on=["prev", "frame"],
            right_on=["id", "frame"],
            suffixes=["", "_prev"],
        ).merge(
            data[["id", "frame", "gender", "x", "y"]],
            left_on=["next", "frame"],
            right_on=["id", "frame"],
            suffixes=["", "_next"],
        )
    ).rename(
        columns={
            "gender_next": "gender_of_next_neighbor",
            "gender_prev": "gender_of_prev_neighbor",
        }
    )

    all_data["distance_to_prev_neighbor"] = np.linalg.norm(
        all_data[["x", "y"]].values - all_data[["x_prev", "y_prev"]].values,
        axis=1,
    )

    all_data["distance_to_next_neighbor"] = np.linalg.norm(
        all_data[["x", "y"]].values - all_data[["x_next", "y_next"]].values,
        axis=1,
    )

    return all_data.drop(columns=["x_prev", "y_prev", "id_prev", "x_next", "y_next", "id_next"])


def calculate_circular_distance_and_gender_vect(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance to the nearest neighbors based on preprocessed neighbor information.

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
        distances_to_prev = np.linalg.norm(id_data[["x", "y"]].values - prev_data[["x", "y"]].values, axis=1)
        gender_of_prev = prev_data["gender"].astype(int).values
        # Calculate distances and genders for next neighbors
        next_data = data[data["id"] == next_id]
        distances_to_next = np.linalg.norm(id_data[["x", "y"]].values - next_data[["x", "y"]].values, axis=1)
        gender_of_next = next_data["gender"].astype(int).values
        # Enhance the data with the new columns
        data.loc[mask, "distance_to_prev_neighbor"] = distances_to_prev
        data.loc[mask, "distance_to_next_neighbor"] = distances_to_next
        data.loc[mask, "gender_of_prev_neighbor"] = gender_of_prev
        data.loc[mask, "gender_of_next_neighbor"] = gender_of_next

    return data


def calculate_circular_distance_and_gender_arc(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distance to the nearest neighbors based on preprocessed neighbor information.

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
            distance, _, _ = sum_distances_between_agents_on_path(p1, p2, middle_path, path_distances)
            distances_to_prev.append(distance)

        gender_of_prev = prev_data["gender"].astype(int).values
        # Calculate distances and genders for next neighbors
        next_data = data[data["id"] == next_id]
        for p1, p2 in zip(id_data[["x", "y"]].values, next_data[["x", "y"]].values):
            distance, _, _ = sum_distances_between_agents_on_path(p1, p2, middle_path, path_distances)
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


def init_gender_code(filename: str) -> str:
    """Generate a code name based on filename."""
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

    return name


def calculate_proximity_analysis(country: str, filename: str, data: pd.DataFrame, fps: int = 25, method: Method = Method.VECT) -> List[Dict[str, Any]]:
    """
    Perform proximity analysis on given data of agents.

    Filtering by frames and categorizing
    by gender proximity for each agent within the selected frames.

    Args:
        country (str): The country code or name related to the dataset.
        filename (str): The name of the file containing the dataset, used to infer gender composition.
        data (pd.DataFrame): A pandas DataFrame witqh rotated data including agent positions and genders.
        fps (int, optional): The frames per second rate used to filter the data. Defaults to 25.
        method (Method): Method of calculation
    Returns:
        List[Dict]: A list of dictionaries, each containing the proximity analysis results for an agent in
        the filtered frames, including country, file name, agent type, ID, frame number, and distances to
        the next and previous neighbors classified by gender similarity.

    Note:
        The function expects 'data' DataFrame to include 'frame', 'id', 'gender',
        'gender_of_next_neighbor', 'gender_of_prev_neighbor', 'distance_to_next_neighbor',
        and 'distance_to_prev_neighbor' columns.
    """
    country_short, country_file = extract_country_info(filename)
    processed_data, fps = process_data_by_method(data, filename, method, country_short, fps)
    proximity_analysis_res = []
    if processed_data.empty:
        print("========")
        print("Processed_data empty", filename)
        print(processed_data)
        print("========")
    frames_to_include = set(range(0, processed_data["frame"].max(), fps))

    # Filter the DataFrame to only include the desired frames
    filtered_data = processed_data[processed_data["frame"].isin(frames_to_include)]

    # Now iterate over the filtered DataFrame
    name = init_gender_code(filename)
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
                "country": country_short,
                "file": country_file,
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


def extract_country_info(filename: str) -> Tuple[str, Path]:
    """Extract country short code and file information."""
    pp = Path(filename).parts
    country_short = Path(pp[-2]).name
    country_file = Path(pp[-2]) / Path(pp[-1]).name
    return country_short, country_file


def process_data_by_method(data: pd.DataFrame, filename: str, method: Method, country_short: str, fps: int) -> Tuple[pd.DataFrame, int]:
    """Process data based on the method selected."""
    if country_short != "pal":
        if method == Method.MERGE:
            processed_data = compute_distance_merge(data)
        elif method == Method.VECT:
            nagents = extract_first_number(filename)
            cleaned_data = filter_frames(data, nagents)
            processed_data = calculate_circular_distance_and_gender_vect(cleaned_data)
        elif method == Method.ARC:
            nagents = extract_first_number(filename)
            cleaned_data = filter_frames(data, nagents)
            processed_data = calculate_circular_distance_and_gender_arc(cleaned_data)
    else:
        processed_data = calculate_circular_distance_and_gender_pal(data)
        fps = 1
    return processed_data, fps



def unpack_and_process(args: Any) -> List[Dict[str, Any]]:
    """Proxy method calling calculate_proximity."""
    return calculate_proximity_analysis(*args)


def prepare_data(country: str, selected_file: str, fps: int, method: Method) -> Tuple[str, str, pd.DataFrame, int, Method]:
    """Load the file and create a trajectory data object."""
    trajectory_data = load_file(selected_file)
    data = trajectory_data.data
    return country, selected_file, data, fps, method


def calculate_with_joblib(init_data: InitData) -> pd.DataFrame:
    """Run calculations with in parallel."""
    tasks = []
    for country in init_data.countries:
        key = Path(country).name
        print(f"prepare tasks: {country} with {key=} and {len(init_data.files[key])} files")

        for filename in init_data.files[key]:
            tasks.append(prepare_data(country, filename, init_data.fps, init_data.method))

    # Define a function to be executed in parallel
    def process_task(task: List[Any]) -> List[Dict[str, Any]]:
        return unpack_and_process(task)

    nproc = -1
    print(f"Running {len(tasks)} taks with {nproc} proc...")
    results = Parallel(n_jobs=nproc)(delayed(process_task)(task) for task in tqdm(tasks))
    # for task in tasks:
    #     process_task(task)

    print(f"Done running tasks in parallel {len(tasks)} tasks ...")
    # Return the final results
    flattened_results = pd.DataFrame(list(itertools.chain.from_iterable(results)))
    flattened_results.to_pickle(init_data.result_csv)
    return flattened_results


def init() -> InitData:
    """
    Initialize the application by setting up the countries, file paths, result CSV path, and FPS (frames per second).

    Returns:
        InitData: A data class containing initialized data including countries, files dictionary, result CSV path, and FPS.
    """
    print("Enter Init")

    method = Method.VECT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_csv = Path(f"{parent_dir}/app_data/proximity_analysis_results_{method}_{timestamp}.csv")
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Created file: {result_csv}")
    fps = 25  # For distance calculations, calculate every fps-frame
    countries = [
        f"{parent_dir}/data/aus",
        f"{parent_dir}/data/ger",
        f"{parent_dir}/data/jap",
        f"{parent_dir}/data/chn",
        f"{parent_dir}/data/pal",
    ]
    files = {}
    for country in countries:
        key = Path(country).name
        files[key] = [str(path) for path in Path(country).glob("*.csv")]

    return InitData(countries=countries, files=files, result_csv=result_csv, fps=fps, method=method)


if __name__ == "__main__":
    init_data = init()
    start_time = time.time()
    proximity_df = calculate_with_joblib(init_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    proximity_df.to_csv(init_data.result_csv, index=False)
