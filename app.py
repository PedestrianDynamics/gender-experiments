import concurrent.futures
import glob
import time
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataclasses import dataclass, field
import itertools
import numpy as np
import pandas as pd
import pedpy
import plotly.express as px
import streamlit as st
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

import analysis as al
import helper as hp
import plots as pl
from anim import animate
from shapely import Polygon, difference

# from memory_profiler import profile
# from memory_profiler import memory_usage


@dataclass
class DataConfig:
    rename_mapping: dict
    column_types: dict
    countries: list
    files: dict = field(default_factory=dict)
    data: dict = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        self.retrieve_files()

    def retrieve_files(self):
        for country in self.countries:
            self.files[country] = glob.glob(f"{country}/*.csv")


def init_session_state(msg):
    if not hasattr(st.session_state, "loaded_data"):
        st.session_state.loaded_data = {}

    if "rotate" not in st.session_state:
        st.session_state["rotate"] = False
        st.session_state["center_x"] = 0.0
        st.session_state["center_y"] = 0.0
        st.session_state["angle_degrees"] = 90.0

    if not hasattr(st.session_state, "config"):
        msg.info("init config")
        st.session_state.config = DataConfig(
            rename_mapping={
                "ID": "id",
                "t(s)": "time",
                "x(m)": "x",
                "y(m)": "y",
            },
            column_types={
                "id": int,
                "gender": int,
                "time": float,
                "x": float,
                "y": float,
            },
            countries=["aus", "ger", "jap", "pal", "chn"],
        )

    if not hasattr(st.session_state, "figures"):
        st.session_state.figures = {}

    if not hasattr(st.session_state, "load_fd_data"):
        st.session_state.load_fd_data = {}


def load_data(msg, country):
    #    for country in st.session_state.config.countries:
    if country not in st.session_state.loaded_data:
        files = st.session_state.config.files[country]

        if not files:
            msg.warning(f"{country}: data missing")
            # continue

        st.write(f"Processing data for <{country}>")
        progress_text = st.empty()
        progress = st.progress(0)
        num_files = len(files)
        st.session_state.loaded_data[country] = []

        for idx, file in enumerate(files):
            data = pd.read_csv(file)
            hp.rename_columns(data, st.session_state.config.rename_mapping)
            set_column_types(data, st.session_state.config.column_types)
            if file == files[0]:
                fps = hp.calculate_fps(data)

            trajectory_data = pedpy.TrajectoryData(data=data, frame_rate=fps)
            st.session_state.loaded_data[country].append(trajectory_data)

            # Update the progress bar
            progress_value = (idx + 1) / num_files
            progress.progress(progress_value)
            progress_text.text(f"File {idx + 1} of {num_files}")


# @st.cache_data


def set_rotation_variables(selected_file, country):
    ger_substrings_to_check = [
        "female",
        "20_19",
        "16_18",
        "40_11",
        "8_16",
        "8_17",
        "32_11",
        "36_11",
        "4_11",
        "4_12",
        "4_13",
        "4_15",
        "4_14",
    ]
    aus_substrings_to_check = ["female", "mix_sorted_40_01", "mix_random_41_01"]
    if country == "ger":
        if any(substring in selected_file for substring in ger_substrings_to_check):
            st.session_state.center_x = 3.1
            st.session_state.center_y = 3
            st.session_state.angle_degrees = 90
        else:
            st.session_state.center_x = 3.1
            st.session_state.center_y = -3
            st.session_state.angle_degrees = 90

    if country == "aus":
        if any(substring in selected_file for substring in aus_substrings_to_check):
            st.session_state.center_x = 1.7
            st.session_state.center_y = -0.2
            st.session_state.angle_degrees = 85
        else:
            st.session_state.center_x = 1.8
            st.session_state.center_y = -6.3
            st.session_state.angle_degrees = 89

    if country == "chn":
        if "female" in selected_file:
            st.session_state.center_x = 0.1
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 87
        elif "mix_random" in selected_file:
            st.session_state.center_x = 0.1
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 85
        else:
            st.session_state.center_x = 0.3
            st.session_state.center_y = 0
            st.session_state.angle_degrees = 90

    if country == "jap":
        st.session_state.center_x = 0
        st.session_state.center_y = 0
        st.session_state.angle_degrees = 0

    if country == "pal":
        st.session_state.center_x = -1.5
        st.session_state.center_y = 0
        st.session_state.angle_degrees = 0


def original(country, selected_file):
    """Plot original data"""
    c1, c2 = st.columns((1, 1))
    # load_data(msg, country)
    msg.write("")
    if country:
        if selected_file:
            # file_index = files.index(selected_file)
            # default values
            set_rotation_variables(selected_file, country)
            # trajectory_data = st.session_state.loaded_data[country][file_index]
            trajectory_data = hp.load_file(selected_file)
            data = trajectory_data.data
            # st.dataframe(data)
            start_time = time.time()
            #        if selected_file not in st.session_state.figures.keys():
            c1, c2 = st.columns((1, 1))
            framerate = c1.slider("Every nth frame", 1, 100, 40, 10)
            ids = data["id"].unique()
            uid = c2.number_input(
                "Insert id of pedestrian",
                value=None,
                min_value=int(min(ids)),
                max_value=int(max(ids)),
                placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
                format="%d",
            )
            rc1, rc2, rc3 = st.columns((1, 1, 1))
            center_x = rc1.number_input(
                "Shift X:",
                value=float(st.session_state["center_x"]),
                step=0.1,
                help="AF=1.7, AM=1.7, CN=0.1, G=3, P=-1.5",
            )
            center_y = rc2.number_input(
                "Shift Y:",
                value=float(st.session_state["center_y"]),
                step=0.1,
                help="AF=0, AM=6.3, CN=0",
            )
            angle_degrees = rc3.number_input(
                "Angle in Degrees:",
                value=st.session_state["angle_degrees"],
                help="A=90, CN=90, G=3",
            )
            if center_x != st.session_state.center_x:
                st.session_state.center_x = center_x

            if center_y != st.session_state.center_y:
                st.session_state.center_y = center_y

            if angle_degrees != st.session_state.angle_degrees:
                st.session_state.angle_degrees = angle_degrees

            fig, do_animate = pl.plot_trajectories(
                data, framerate, uid, exterior, interior
            )
            st.plotly_chart(fig)
            # neighborhood
            if len(ids) > 2 and country != "pal":
                fig = hp.plot_neighbors_analysis(data, ids, exterior, interior)
                st.plotly_chart(fig)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken to plot trajectories: {elapsed_time:.2f} seconds")

            if do_animate:
                rotated_data = hp.rotate_trajectories(
                    data,
                    st.session_state.center_x,
                    st.session_state.center_y,
                    st.session_state.angle_degrees,
                )
                rotated_trajectory_data = pedpy.TrajectoryData(
                    rotated_data, trajectory_data.frame_rate
                )
                anm = animate(
                    rotated_trajectory_data,
                    walkable_area,
                    width=500,
                    height=500,
                    every_nth_frame=100,
                    radius=0.1,  # 0.75
                    title_note="(<span style='color:green;'>M</span>, <span style='color:blue;'>F</span>)",
                )
                st.plotly_chart(anm)
                st.dataframe(rotated_trajectory_data.data)


def density_speed_time_series(country, file, fps, dv, diff_const):
    set_rotation_variables(file, country)
    trajectory_data = hp.load_file(file)
    data = trajectory_data.data
    rotated_data = hp.rotate_trajectories(
        data,
        st.session_state.center_x,
        st.session_state.center_y,
        st.session_state.angle_degrees,
    )

    with st.spinner(f"Calculating {country} ..."):
        start_time = time.time()
        density = al.calculate_instantaneous_density_per_frame(rotated_data, fps)
        speed = al.calculate_speed(rotated_data, dv)
        steady_state_index = al.calculate_steady_state(
            speed["speed"], window_size=5, threshold=0.1, diff_const=diff_const
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.info(f"Time taken to calculate density: {elapsed_time:.2f} seconds")
        fig = pl.plot_time_series(
            density, speed, trajectory_data.frame_rate, steady_state_index
        )
        st.plotly_chart(fig)


def fundamental_diagram(country, fps, dv, diff_const):
    with st.spinner(f"Calculating {country} ..."):
        start_time = time.time()
        mean_density = []
        mean_speed = []
        for file in st.session_state.config.files[country]:
            set_rotation_variables(file, country)
            trajectory_data = hp.load_file(file)
            data = trajectory_data.data
            rotated_data = hp.rotate_trajectories(
                data,
                st.session_state.center_x,
                st.session_state.center_y,
                st.session_state.angle_degrees,
            )

            density = al.calculate_instantaneous_density_per_frame(rotated_data, fps)

            speed = al.calculate_speed(rotated_data, dv)
            steady_state_index = al.calculate_steady_state(
                speed["speed"], window_size=5, threshold=0.1, diff_const=diff_const
            )
            mean_speed.append(np.mean(speed["speed"].iloc[steady_state_index:]))
            mean_density.append(np.mean(density["instantaneous_density"]))

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.info(f"Time taken to calculate density: {elapsed_time:.2f} seconds")
        mean_density = np.array(mean_density)
        mean_speed = np.array(mean_speed)
        return mean_speed, mean_density


def calculate_proximity_analysis(country, rotated_data):
    processed_data = al.calculate_circular_distance_and_gender(rotated_data)
    proximity_analysis_res = []
    fps = 25
    frames_to_include = set(range(0, processed_data["frame"].max(), fps))

    # Filter the DataFrame to only include the desired frames
    filtered_data = processed_data[processed_data["frame"].isin(frames_to_include)]

    # Now iterate over the filtered DataFrame
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

    return proximity_analysis_res


def unpack_and_process(args):
    return calculate_proximity_analysis(*args)


def prepare_data(country, selected_file):
    set_rotation_variables(selected_file, country)
    trajectory_data = hp.load_file(selected_file)
    data = trajectory_data.data
    rotated_data = hp.rotate_trajectories(
        data,
        st.session_state.center_x,
        st.session_state.center_y,
        st.session_state.angle_degrees,
    )
    return country, rotated_data


def calculate_with_progress():
    # Prepare tasks
    tasks = []
    for country in st.session_state.config.countries:
        with st.spinner(f"Preparing tasks for {country}"):
            for file in st.session_state.config.files[country]:
                tasks.append(prepare_data(country, file))

    with st.spinner("Running..."):
        # Create a progress bar
        progress_bar = st.progress(0)

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

                # Update progress bar
                progress_bar.progress(i / len(tasks))

    # Return the final results
    return results


def calculate_with_singular():
    # Prepare tasks
    tasks = []
    for country in st.session_state.config.countries:
        with st.spinner(f"Preparing tasks for {country}"):
            for file in st.session_state.config.files[country][0:2]:
                tasks.append(prepare_data(country, file))

    # tasks = [
    #     prepare_data(country, file)
    #     for country in st.session_state.config.countries
    #     for file in st.session_state.config.files[country]
    # ]

    with st.spinner("Running..."):
        # Create a progress bar
        progress_bar = st.progress(0)
        results = []
        for i, task in enumerate(tasks):
            result = unpack_and_process(task)
            results.append(result)
            # Update progress bar
            progress_bar.progress(i / len(tasks))

    # Return the final results
    return results


def calculate_with_joblib():
    # Prepare tasks
    tasks = []
    for country in st.session_state.config.countries:
        with st.spinner(f"Preparing tasks for {country}"):
            for file in st.session_state.config.files[country][0:1]:
                tasks.append(prepare_data(country, file))

    # Define a function to be executed in parallel
    def process_task(task):
        return unpack_and_process(task)

    st.info(f"Running tasks in parallel {len(tasks)} ...")
    results = Parallel(n_jobs=-1)(
        delayed(process_task)(task) for task in tqdm(tasks, desc="Processing")
    )

    return results


# Main
if __name__ == "__main__":
    msg = st.empty()
    st.sidebar.title("Trajectory Visualization")
    exterior, interior = hp.generate_parcour()
    walkable_area = pedpy.WalkableArea(difference(Polygon(exterior), Polygon(interior)))
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "View trajectories",
            "Fundamental diagram",
            "Proximity analysis",
            "write enhanced data",
        ]
    )

    init_session_state(msg)
    country = st.sidebar.selectbox(
        "Select a country:", st.session_state.config.countries
    )
    files = st.session_state.config.files[country]
    n_female, n_male, n_mixed_random, n_mixed_sorted = hp.get_numbers_country(country)
    st.sidebar.info(
        f" Number files: {len(files)}\n- Female files: {n_female}\n- Male files: {n_male}\n- Mix_sorted files: {n_mixed_sorted}\n- Mix random files: {n_mixed_random}"
    )

    file_names = [f.split("/")[-1] for f in files]
    sorted_file_names = sorted(file_names, key=hp.sorting_key)
    selected_file = st.sidebar.radio(
        "Select a file", sorted_file_names, horizontal=True
    )
    selected_file = country + "/" + selected_file

    # with tab0:
    #     st.header("Summary of the data")
    #     generate = st.checkbox("Generate statistics", value=False)
    #     if generate:
    #         st.markdown(
    #             f"Number of countries: {len(st.session_state.config.countries)}"
    #         )
    #         for country in st.session_state.config.countries:
    #             st.info(f"Country: {country}")
    #             files = st.session_state.config.files[country]
    #             st.markdown(f"- Number of files {len(files)}")

    with tab1:
        #    mem_usage_before = memory_usage()[0]
        original(country, selected_file)
        #   mem_usage_after = memory_usage()[0]
        #  st.info(f"Memory usage: {mem_usage_after - mem_usage_before:.2f} MiB")

    with tab2:
        do_calculations = st.checkbox(
            "Calculate density vs time, fundamental diagram, ...", value=False
        )
        c0, c1, c2 = st.columns((1, 1, 1))
        if do_calculations:
            calculations = c0.radio(
                "Choose calculation",
                ["micro_fd_rudina", "time_series", "fundamental_diagram"],
            )
            if calculations == "micro_fd_rudina":
                countries = c1.multiselect("Country", st.session_state.config.countries)
                fps = c2.slider("fps", 25, 500, 100, 25, help="skip so many points")
                fig = pl.plot_rudina_fd(countries, fps)
                st.plotly_chart(fig)
            else:
                fps = c1.slider(
                    "fps",
                    1,
                    100,
                    25,
                    5,
                    help="jump every fps frame for density calculation",
                )
                dv = c2.slider(
                    "steps",
                    1,
                    100,
                    10,
                    5,
                    help="number of frames to jump for diff speed",
                )
                diff_const = c1.slider(
                    "diff_const", 1, 500, 5, 1, help="window steady state"
                )

            if calculations == "time_series":
                density_speed_time_series(country, selected_file, fps, dv, diff_const)

            if calculations == "fundamental_diagram":
                all_data = {}
                for country in st.session_state.config.countries:
                    mean_speed, mean_density = fundamental_diagram(
                        country, fps, dv, diff_const
                    )
                    fig = pl.plot_fundamental_diagram(country, mean_density, mean_speed)
                    st.plotly_chart(fig)
                    all_data[country] = (mean_speed, mean_density)

                fig = pl.plot_fundamental_diagram_all(all_data)
                st.plotly_chart(fig)

    with tab3:
        # do_analysis = st.checkbox("Perform gender analysis", value=False)
        do_analysis = st.radio(
            "Choose option",
            ["voronoi", "calculate_gender_analysis", "plot_existing_data"],
        )
        if do_analysis == "voronoi":
            pass
        if do_analysis == "calculate_gender_analysis":
            start_time = time.time()
            proximity_analysis_results = calculate_with_progress()
            # proximity_analysis_results = calculate_with_joblib()
            # proximity_analysis_results = calculate_with_singular()
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.info(f"Time taken: {elapsed_time:.2f} seconds")
            flattened_results = list(
                itertools.chain.from_iterable(proximity_analysis_results)
            )
            proximity_df = pd.DataFrame(flattened_results)
            proximity_df.to_csv("proximity_analysis_results.csv", index=False)

            st.dataframe(proximity_df)
            with st.spinner("Calculating T-tests ..."):
                same_gender_distances_next = proximity_df[
                    "same_gender_proximity_next"
                ].dropna()
                diff_gender_distances_next = proximity_df[
                    "diff_gender_proximity_next"
                ].dropna()
                same_gender_distances_prev = proximity_df[
                    "same_gender_proximity_prev"
                ].dropna()
                diff_gender_distances_prev = proximity_df[
                    "diff_gender_proximity_prev"
                ].dropna()

                # Perform a T-test
                t_stat_next, p_val_next = stats.ttest_ind(
                    same_gender_distances_next, diff_gender_distances_next
                )
                t_stat_prev, p_val_prev = stats.ttest_ind(
                    same_gender_distances_prev, diff_gender_distances_prev
                )

                st.info(
                    f"T-Test results proximity_next: T-Statistic = {t_stat_next:.03f}, P-Value = {p_val_next:.03f}"
                )
                st.info(
                    f"T-Test results proximity_prev: T-Statistic = {t_stat_next:.03f}, P-Value = {p_val_prev:.03f}"
                )

            proximity_melted = proximity_df.melt(
                id_vars=["id", "frame", "country"],
                value_vars=[
                    "same_gender_proximity_next",
                    "diff_gender_proximity_next",
                    "same_gender_proximity_prev",
                    "diff_gender_proximity_prev",
                ],
                var_name="category",
                value_name="distance",
            )

            # Creating a box plot
            fig = px.box(
                proximity_melted,
                x="category",
                y="distance",
                color="country",
                title="Proximity Analysis Based on Gender and Country",
                labels={"distance": "Proximity Distance", "category": "Category"},
            )

            fig.update_layout(
                yaxis_title="Distance",
                xaxis_title="Gender Proximity Category",
                showlegend=True,
            )

            st.plotly_chart(fig)

    with tab4:
        convert = st.checkbox("Convert data", value=False)
        if convert:
            k = 3
            for country in st.session_state.config.countries:
                if country == "pal":
                    continue
                files = st.session_state.config.files[country]
                with st.spinner(f"Converting files for {country} ..."):
                    for selected_file in files:
                        st.info(f"{country}, {selected_file}")
                        trajectory_data = hp.load_file(selected_file)
                        data = trajectory_data.data
                        rotated_data = hp.rotate_trajectories(
                            data,
                            st.session_state.center_x,
                            st.session_state.center_y,
                            st.session_state.angle_degrees,
                        )
                        first_frame = rotated_data["frame"].to_numpy()[0]
                        # st.info(first_frame)
                        nearest_dist, nearest_ind = hp.get_neighbors_at_frame(
                            first_frame, rotated_data, k
                        )
                        agents = np.unique(rotated_data["id"])
                        st.info(agents)
                        for agent in agents:
                            (
                                neighbors,
                                neighbors_ids,
                                area,
                                agent_distances,
                                neighbor_type,
                            ) = hp.get_neighbors_special_agent_data(
                                agent,
                                first_frame,
                                rotated_data,
                                nearest_dist,
                                nearest_ind,
                            )
                            st.info(
                                f"{agent=}, neighbor {neighbors_ids[0]}, type {neighbor_type}"
                            )
                        break
