import concurrent.futures
import pedpy
import time
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import analysis as al
import glob
import helper as hp
from tqdm import tqdm

from joblib import Parallel, delayed


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


def calculate_proximity_analysis(country, rotated_data):
    processed_data = al.calculate_circular_distance_and_gender(rotated_data)
    proximity_analysis_res = []
    fps = 25
    frames_to_include = set(range(0, processed_data["frame"].max(), fps))

    # Filter the DataFrame to only include the desired frames
    filtered_data = processed_data[processed_data["frame"].isin(frames_to_include)]

    # Now iterate over the filtered DataFrame
    
    with tqdm(total=len(filtered_data)) as pbar:
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
                    "id": row["id"],
                    "frame": row["frame"],
                    "same_gender_proximity_next": same_gender_proximity_next,
                    "diff_gender_proximity_next": diff_gender_proximity_next,
                    "same_gender_proximity_prev": same_gender_proximity_prev,
                    "diff_gender_proximity_prev": diff_gender_proximity_prev,
                }
            )
            pbar.update(1)

    return proximity_analysis_res


def unpack_and_process(args):
    return calculate_proximity_analysis(*args)


def prepare_data(country, selected_file):
    trajectory_data = load_file(selected_file)
    data = trajectory_data.data
    return country, data


def calculate_with_progress(countries, files):
    res_file = "proximity_results"
    # res_file_path = Path(res_file)
    # if res_file_path.exists():
    #     st.info("Found ")
    #     return pd.read_pickle(res_file)

    # Prepare tasks
    tasks = []

    for country in countries:
        print(f"prepare tasks: {country}")
        for f in files[country]:
            tasks.append(prepare_data(country, f))

    with tqdm(total=len(tasks)) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(unpack_and_process, task): task for task in tasks
            }

            results = []
            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_task), 1
            ):
                # Result from the completed task
                result = future.result()
                results.append(result)
                # Update the progress bar
                pbar.update(1)

    # Return the final results
    flattened_results = list(itertools.chain.from_iterable(results))
    flattened_results = pd.DataFrame(flattened_results)
    flattened_results.to_pickle(res_file)
    return flattened_results


def calculate_with_joblib(countries, files):
    # Prepare tasks
    res_file = "proximity_results"

    tasks = []
    for country in countries[0:2]:
        print(f"prepare tasks: {country}")
        for file in files[country][0:2]:
            tasks.append(prepare_data(country, file))

    # Define a function to be executed in parallel
    def process_task(task):
        return unpack_and_process(task)

    print(f"Running tasks in parallel {len(tasks)} ...")
    results = Parallel(n_jobs=-1)(
        delayed(process_task)(task) for task in tqdm(tasks, desc="Processing")
    )

    # Return the final results
    flattened_results = list(itertools.chain.from_iterable(results))
    flattened_results = pd.DataFrame(flattened_results)
    flattened_results.to_pickle(res_file)

    return flattened_results


def init():
    result_csv = Path("proximity_analysis_results.csv")
    countries = [
        "aus",
        "ger",
        "jap",
        "chn",
        "pal",
    ]
    files = {}
    for country in countries:
        files[country] = glob.glob(f"{country}/*.csv")

    return countries, files, result_csv


if __name__ == "__main__":
    countries, files, result_csv = init()
    start_time = time.time()
    proximity_df = calculate_with_joblib(countries, files)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    proximity_df.to_csv(result_csv, index=False)
