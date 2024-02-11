"""Main entry point to the app."""

import concurrent.futures
import glob
import itertools
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pedpy
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# from joblib import Parallel, delayed
from scipy import stats
from shapely import Polygon, difference

import analysis as al
import helper as hp
import plots as pl
from anim import animate

# global variables for csv proximity file
proximity_results = {}
proximity_results["path"] = Path("proximity_analysis_results.csv")
proximity_results["url"] = "https://fz-juelich.sciebo.de/s/U5rujIKIaZenIUg/download"
path = Path(__file__)
ROOT_DIR = path.parent.absolute()


# from plotly.subplots import make_subplots
# import plotly.graph_objects as go


# from tqdm import tqdm


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
        """Initialize the DataConfig instance by retrieving files for each country."""
        self.retrieve_files()

    def retrieve_files(self) -> None:
        """Retrieve the files for each country specified in the countries list.

        The files are expected to be CSV files located in directories named after the countries.

        This method updates the `files` dictionary with country names as keys and lists
        of file paths as values.
        """
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
            countries=[
                # "enhanced_aus",
                # "enhanced_ger",
                # "enhanced_jap",
                # "enhanced_chn",
                "aus",
                "ger",
                "jap",
                "chn",
                "pal",
            ],
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
            hp.set_column_types(data, st.session_state.config.column_types)
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
    """First tab. Plot original data, animatoin, neighborhood."""
    c1, c2 = st.columns((1, 1))
    do_rotate = False
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
            rc0, rc1, rc2, rc3 = st.columns((1, 1, 1, 1))
            st.write("---------")
            columns_to_display = ["id", "frame", "time", "x", "y", "prev", "next"]
            display = rc0.checkbox("Data", value=True, help="Display data table")
            if display:
                if country != "pal":
                    st.dataframe(trajectory_data.data.loc[:, columns_to_display])
                else:
                    st.warning("For pal there are no neighbors.")
            do_plot_trajectories = rc1.checkbox(
                "Plot", value=False, help="Plot trajectories"
            )

            do_animate = rc2.checkbox(
                "Animation", value=False, help="Visualise movement of trajecories"
            )
            get_neighborhood = rc3.checkbox(
                "Neighbors", value=False, help="Calculate and visualize neighbors"
            )
            ids = data["id"].unique()
            if do_plot_trajectories:
                # do_fix = c1.checkbox("Fix", value=False)
                # if do_fix:
                #     set_rotation_variables(selected_file, country)
                #     shift_y = st.number_input(
                #         "shift y",
                #         value=-6.0,
                #         min_value=-10.0,
                #         max_value=10.0,
                #     )
                #     shift_x = st.number_input(
                #         "shift x",
                #         value=0.0,  # st.session_state.center_x,
                #         min_value=-10.0,
                #         max_value=10.0,
                #     )

                #     angle = st.number_input(
                #         "angle",
                #         value=0,  # st.session_state.angle_degrees,
                #         min_value=-90,
                #         max_value=90,
                #     )
                #     rotated_data = hp.rotate_trajectories(data, shift_x, shift_y, angle)
                #     data = rotated_data

                c1, c2, c3 = st.columns((1, 1, 1))
                plot_parcour = c1.checkbox("Parcour", value=True)
                framerate = c2.slider("Every nth frame", 1, 100, 40, 10)

                uid = c3.number_input(
                    "Insert id of pedestrian",
                    value=None,
                    min_value=int(min(ids)),
                    max_value=int(max(ids)),
                    placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
                    format="%d",
                )

                fig = pl.plot_trajectories(
                    data, framerate, uid, exterior, interior, plot_parcour
                )
                # if do_fix:
                #     write_to_file = c1.checkbox("Write to file", value=False)
                #     if write_to_file:
                #         newfile = f"enhanced_{selected_file}"
                #         st.warning(newfile)
                #         rename_mapping = {
                #             "id": "ID",
                #             "time": "t(s)",
                #             "x": "x(m)",
                #             "y": "y(m)",
                #         }

                #         data.rename(columns=rename_mapping, inplace=True)
                #         selected_columns = [
                #             "ID",
                #             "next",
                #             "prev",
                #             "gender",
                #             "frame",
                #             "t(s)",
                #             "x(m)",
                #             "y(m)",
                #         ]
                #         data[selected_columns].to_csv(newfile, index=False)
                #         st.info("Done!")

                st.plotly_chart(fig)
            # neighborhood
            if get_neighborhood and len(ids) > 2 and country != "pal":
                fig = hp.plot_neighbors_analysis(
                    data, ids, exterior, interior, do_rotate
                )
                st.plotly_chart(fig)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken to plot trajectories: {elapsed_time:.2f} seconds")

            if do_animate:
                if do_rotate:
                    rotated_data = hp.rotate_trajectories(
                        data,
                        st.session_state.center_x,
                        st.session_state.center_y,
                        st.session_state.angle_degrees,
                    )

                    rotated_trajectory_data = pedpy.TrajectoryData(
                        rotated_data, trajectory_data.frame_rate
                    )
                else:
                    rotated_trajectory_data = pedpy.TrajectoryData(
                        data, trajectory_data.frame_rate
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

    return do_rotate


def density_speed_time_series_macro(country, file, fps, dv, diff_const):
    """Calculate density and speed series using Corbetta's method for density."""
    set_rotation_variables(file, country)
    trajectory_data = hp.load_file(file)
    data = trajectory_data.data
    rotated_data = data

    with st.spinner(f"Calculating {country} ..."):
        start_time = time.time()
        density = al.calculate_instantaneous_density_per_frame(rotated_data, fps)  #
        speed = al.calculate_speed(rotated_data, dv)
        steady_state_index = al.calculate_steady_state(
            speed["speed"], window_size=5, threshold=0.1, diff_const=diff_const
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.info(f"Time taken to calculate density macro: {elapsed_time:.2f} seconds")
        fig = pl.plot_time_series(  #
            density,
            speed,
            trajectory_data.frame_rate,
            steady_state_index,
            "instantaneous_density",
        )
        st.plotly_chart(fig)


def density_speed_time_series_micro(country, file, fps, dv, diff_const):
    """Calculate the individual density (Voronoi 1D)."""
    msg = st.empty()
    trajectory_data = hp.load_file(file)
    rotated_data = trajectory_data.data
    result_csv = proximity_results["path"]

    if not result_csv.exists():
        msg.warning(f"{result_csv} does not exist yet!")
        with st.spinner("Downloading ..."):
            hp.download_csv(proximity_results["url"], proximity_results["path"])

    if result_csv.exists():
        msg.info(f"Reading file {result_csv}")
        df = pd.read_csv(result_csv)
    cs, cd = st.columns((1, 1))
    with st.spinner(f"Calculating {country} ..."):
        start_time = time.time()
        filter_df = df[(df["country"] == country) & (df["file"] == file)]
        st.info(f"{file}")
        st.dataframe(filter_df)
        density = al.calculate_individual_density_csv(filter_df)
        st.info(f"Density for {file}")
        cd.dataframe(density)
        msg.info("calculate speed")
        speed = al.calculate_speed(rotated_data, dv)

        cs.dataframe(speed.loc[:, ["frame", "id", "speed"]])
        msg.info("calculate speed steady state")
        steady_state_index = al.calculate_steady_state(
            speed["speed"], window_size=5, threshold=0.1, diff_const=diff_const
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        msg.info(
            f"Time taken to calculate speed and density micro: {elapsed_time:.2f} seconds"
        )
        fig = pl.plot_time_series(
            density,
            speed,
            trajectory_data.frame_rate,
            steady_state_index,
            "individual_density",
        )
        st.plotly_chart(fig)


def fundamental_diagram(country, fps, dv, diff_const, do_rotate, method: str):
    msg = st.empty()
    result_csv = proximity_results["path"]
    if not result_csv.exists():
        msg.warning(f"{result_csv} does not exist yet!")
        with st.spinner("Downloading ..."):
            hp.download_csv(proximity_results["url"], proximity_results["path"])

    if result_csv.exists():
        msg.info(f"Reading file {result_csv}")
        df = pd.read_csv(result_csv)

    with st.spinner(f"Calculating {country} ..."):
        start_time = time.time()
        mean_density = []
        mean_speed = []
        density_columns = []
        speed_columns = []
        density_key = (
            "instantaneous_density" if method == "macro" else "individual_density"
        )

        for file in st.session_state.config.files[country]:
            msg.info(f"{file}")
            # set_rotation_variables(file, country)
            trajectory_data = hp.load_file(file)
            data = trajectory_data.data
            if do_rotate:
                rotated_data = hp.rotate_trajectories(
                    data,
                    st.session_state.center_x,
                    st.session_state.center_y,
                    st.session_state.angle_degrees,
                )
            else:
                rotated_data = data

            if method == "macro":
                density = al.calculate_instantaneous_density_per_frame(
                    rotated_data, fps
                )

            elif method == "micro":
                msg.info("start calculating")
                filter_df = df[(df["country"] == country) & (df["file"] == file)]
                density = al.calculate_individual_density_csv(filter_df)  #

                start = time.time()
                # density = al.calculate_individual_density(rotated_data)  #
                end = time.time()
                msg.info(f"Finished with {file} in {end-start} s")
            else:
                st.warning(
                    f"Method {method} is not implemented. Choose between macro or micro"
                )
                st.stop()
            speed = al.calculate_speed(rotated_data, dv)  #
            steady_state_index = al.calculate_steady_state(
                speed["speed"], window_size=5, threshold=0.1, diff_const=diff_const
            )
            # mean_speed.append(np.mean(speed["speed"].iloc[steady_state_index:]))
            # mean_density.append(np.mean(density[density_key]))
            # mean_speed.append(speed["speed"].iloc[steady_state_index:])
            # mean_density.append(density[density_key])
            density_columns.append(density[density_key])
            speed_columns.append(speed["speed"])

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.info(f"Time taken to calculate density: {elapsed_time:.2f} seconds")

        density_series = pd.concat(density_columns, ignore_index=True)
        density_array = density_series.to_numpy()
        speed_series = pd.concat(speed_columns, ignore_index=True)
        speed_array = speed_series.to_numpy()

        mean_density = np.array(mean_density)
        mean_speed = np.array(mean_speed)
        # return mean_speed, mean_density
        np.save(f"density_array_{country}.npy", density_array)
        np.save(f"speed_array_{country}.npy", speed_array)

        return speed_array, density_array


# Main
if __name__ == "__main__":
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/PedestrianDynamics/gender-experiments"
    repo_name = f"[![Repo]({gh})]({repo})"
    c1, c2 = st.sidebar.columns((1.2, 0.5))
    c2.markdown(repo_name, unsafe_allow_html=True)
    c1.write(
        "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7697604.svg)](https://doi.org/10.5281/zenodo.7697604)"
    )
    st.sidebar.image(f"{ROOT_DIR}/logo.png", use_column_width=True)

    msg = st.empty()
    # st.sidebar.title("Trajectory Visualization")
    c1, c2 = st.sidebar.columns((1.8, 0.2))
    flag = c2.empty()
    exterior, interior = hp.generate_parcour()
    walkable_area = pedpy.WalkableArea(difference(Polygon(exterior), Polygon(interior)))
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👫🏻 View trajectories",
            "📉 Fundamental diagram",
            "📍 Proximity analysis",
            "📂 write enhanced data",
        ]
    )

    init_session_state(msg)
    country = c1.selectbox("Select a country:", st.session_state.config.countries)
    if "jap" in country:
        flag.write(":flag-jp:")
    if "aus" in country:
        flag.write(":flag-ac:")
    if "chn" in country:
        flag.write(":flag-cn:")
    if "ger" in country:
        flag.write(":flag-de:")
    if "pal" in country:
        flag.write(":flag-ae:")

    files = st.session_state.config.files[country]
    n_female, n_male, n_mixed_random, n_mixed_sorted = hp.get_numbers_country(country)
    st.sidebar.info(
        f" Number files: {len(files)}\n- Female files: {n_female}\n- Male files: {n_male}\n- Mix sorted files: {n_mixed_sorted}\n- Mix random files: {n_mixed_random}"
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
        do_rotate = original(country, selected_file)
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
                [
                    "micro_fd_rudina",
                    "time_series_macro",
                    "FD_macro",
                    "time_series_micro",
                    "FD_micro",
                ],
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

            if calculations == "time_series_macro":
                density_speed_time_series_macro(  #
                    country, selected_file, fps, dv, diff_const
                )
            if calculations == "time_series_micro":
                density_speed_time_series_micro(
                    country, selected_file, fps, dv, diff_const
                )

            if calculations == "FD_macro":
                method = calculations.split("_")[-1]
                st.info(method)
                all_data = {}
                for country in st.session_state.config.countries:
                    file_density = Path(f"density_array_{country}.npy")
                    file_speed = Path(f"speed_array_{country}.npy")
                    if file_density.exists() and file_speed.exists():
                        mean_density = np.load(file_density)
                        mean_speed = np.load(file_speed)
                    else:
                        mean_speed, mean_density = fundamental_diagram(  #
                            country, fps, dv, diff_const, do_rotate, method
                        )

                    fig = pl.plot_fundamental_diagram(country, mean_density, mean_speed)
                    st.plotly_chart(fig)
                    all_data[country] = (mean_speed, mean_density)

                fig = pl.plot_fundamental_diagram_all(all_data)
                st.plotly_chart(fig)

            if calculations == calculations == "FD_micro":
                all_data = {}
                result_csv = proximity_results["path"]
                # Path("proximity_analysis_results.csv")
                if not result_csv.exists():
                    msg.warning(f"{result_csv} does not exist yet!")

                    with st.spinner("Downloading ..."):
                        hp.download_csv(
                            proximity_results["url"], proximity_results["path"]
                        )

                if result_csv.exists():
                    msg.info(f"Reading file {result_csv}")
                    df = pd.read_csv(result_csv)
                st.divider()
                for country in st.session_state.config.countries:
                    if country == "pal":
                        continue

                    filter_df = df[df["country"] == country]
                    if "jap" == country:
                        flag = ":flag-jp:"
                    if "aus" == country:
                        flag = ":flag-ac:"
                    if "chn" == country:
                        flag = ":flag-cn:"
                    if "ger" == country:
                        flag = ":flag-de:"

                    st.info(f"Country: {country} {flag}")
                    st.dataframe(filter_df)
                    density = al.calculate_individual_density_csv(filter_df)
                    st.info("Density")
                    st.dataframe(density)
                    #     fig = pl.plot_fundamental_diagram(country, mean_density, mean_speed)
                    #     st.plotly_chart(fig)
                    # all_data[country] = (mean_speed, mean_density)

                # fig = pl.plot_fundamental_diagram_all(all_data)
                hp.show_fig(fig, html=True)

    with tab3:
        # do_analysis = st.checkbox("Perform gender analysis", value=False)
        do_analysis = st.radio(
            "Choose option",
            [
                "voronoi",
                "load_gender_analysis",
                "calculate_gender_analysis",
                "plot_existing_data",
            ],
        )
        if do_analysis == "voronoi":
            pass

        if do_analysis == "calculate_gender_analysis":
            start_time = time.time()

            with st.spinner("Calculating ..."):
                result = subprocess.run(
                    ["python", "proximity.py"], capture_output=True, text=True
                )
                # if result.stdout:
                #     st.write(result.stdout)
                if result.stderr:
                    st.error(result.stderr)

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.info(f"Running time: {elapsed_time/60:.2} min")

        if do_analysis == "load_gender_analysis":
            result_csv = Path("proximity_analysis_results.csv")
            msg = st.empty()
            c0, c1, c2, c3 = st.columns((1, 1, 1, 1))
            st.divider()

            if not result_csv.exists():
                msg.warning(f"{result_csv} does not exist yet!")
                csv_url = "https://fz-juelich.sciebo.de/s/U5rujIKIaZenIUg/download"
                with st.spinner("Downloading ..."):
                    hp.download_csv(csv_url, result_csv)

            if result_csv.exists():
                msg.info(f"Reading file {result_csv}")
                proximity_df = pd.read_csv(result_csv)
            else:
                msg.warning(f"File {result_csv} not found.")
                st.stop()

            msg.empty()

            show_dataframe = c0.checkbox("Show data", value=True)
            ttest = c1.checkbox("T-test", value=False)
            plot_pdf = c2.checkbox("PDF", value=False)
            debug = c3.checkbox(
                "Time-series",
                value=False,
                help="Plot times series of distance per file",
            )
            if show_dataframe:
                st.info("All proximity data")
                st.dataframe(proximity_df)

            if debug:
                st.info(f"Data for {selected_file}")
                st.dataframe(proximity_df.loc[proximity_df["file"] == selected_file])
                ids = proximity_df["id"].unique()
                uid = st.number_input(
                    "Insert id of pedestrian",
                    value=int(min(ids)),
                    min_value=int(min(ids)),
                    max_value=int(max(ids)),
                    placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
                    format="%d",
                )
                condition = (
                    (proximity_df["id"] == uid)
                    & (proximity_df["file"] == selected_file),
                )
                field_name = {
                    "same_gender_proximity_next": "same next",
                    "same_gender_proximity_prev": "same  prev",
                    "diff_gender_proximity_next": "diff next",
                    "diff_gender_proximity_prev": "diff prev",
                }
                colors = ["blue", "crimson", "green", "magenta"]
                fig = go.Figure()
                for (field_, name), color in zip(field_name.items(), colors):
                    frames = proximity_df.loc[
                        condition,
                        "frame",
                    ]
                    # y_col = proximity_df.loc[proximity_df["id"] == uid, field]
                    y_col = proximity_df.loc[condition, field_]
                    title_text = rf"$ \text{ {name} }\in [{np.min(y_col):.2f}, {np.max(y_col):.2f}], \mu = {np.mean(y_col):.2f} \pm {np.std(y_col):.2f}$"
                    if not np.all(np.isnan(y_col)):
                        fig.add_trace(
                            go.Scatter(
                                x=frames,
                                y=y_col,
                                marker=dict(color=color),
                                mode="lines",
                                name=title_text,
                                showlegend=True,
                            ),
                        )

                fig.update_layout(
                    # title=title_text,
                    xaxis_title="frame",
                    yaxis_title=r"Distance / m",
                    # xaxis=dict(scaleanchor="y"),  # , range=[xmin, xmax]),
                    yaxis=dict(scaleratio=1, range=[0, 6]),
                    showlegend=True,
                )
                st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=500)

            if ttest:
                st.divider()
                st.markdown(
                    ":information_source: **The mean distance between pairs of the same gender is equal to the mean distance between pairs of different genders.**"
                )
                st.latex("p <= 0.05 \\rightarrow \\text{ reject}\; H_0")

                for country in ["aus", "ger", "jap", "chn"]:
                    msg = f"{country}\n"
                    filtered_data = proximity_df[
                        (proximity_df["country"] == country)
                        & (proximity_df["type"] == "mix_sorted")
                    ]
                    st.dataframe(filtered_data)
                    with st.spinner("Calculating T-tests ..."):
                        same_gender_distances_next = filtered_data[
                            "same_gender_proximity_next"
                        ].dropna()
                        diff_gender_distances_next = filtered_data[
                            "diff_gender_proximity_next"
                        ].dropna()
                        same_gender_distances_prev = filtered_data[
                            "same_gender_proximity_prev"
                        ].dropna()
                        diff_gender_distances_prev = filtered_data[
                            "diff_gender_proximity_prev"
                        ].dropna()

                        # Perform a T-test
                        t_stat_next, p_val_next = stats.ttest_ind(
                            same_gender_distances_next, diff_gender_distances_next
                        )
                        t_stat_prev, p_val_prev = stats.ttest_ind(
                            same_gender_distances_prev,
                            diff_gender_distances_prev,
                        )

                        msg += f"- Next neighbors: T-Statistic = {t_stat_next:.02f}, P-Value = {p_val_next:.02f}\n"
                        msg += f"- Prev neighbors: T-Statistic = {t_stat_prev:.02f}, P-Value = {p_val_prev:.02f}"
                        st.info(msg)

            if plot_pdf:
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

                proximity_melted["distance"] = proximity_melted["distance"].fillna(0)
                # st.info("proximity_melted")
                # st.dataframe(proximity_melted)
                c1, _, c2 = st.columns((1, 0.5, 1))
                type_name = {
                    "same_gender_proximity_next": "same next",
                    "same_gender_proximity_prev": "same  prev",
                    "diff_gender_proximity_next": "diff next",
                    "diff_gender_proximity_prev": "diff prev",
                }
                for country in ["aus", "ger", "jap", "chn"]:

                    fig = make_subplots(
                        rows=1,
                        cols=1,
                        subplot_titles=[f"<b>{country}</b>"],
                        x_title="Distance / m",
                        y_title="PDF",
                    )
                    msg = f"{country}\n"
                    for (type_, name), color in zip(
                        type_name.items(), ["blue", "crimson", "green", "magenta"]
                    ):

                        line_property = "solid" if type_.endswith("next") else "dash"
                        if line_property == "dash":
                            if type_.startswith("diff"):
                                line_property = "dot"

                        filtered_data = proximity_melted.loc[
                            (proximity_melted["country"] == country)
                            & (proximity_melted["category"] == type_)
                        ]
                        filtered_data = filtered_data.loc[
                            filtered_data["distance"] != 0
                        ]

                        distances = np.unique(filtered_data["distance"])
                        loc = distances.mean()
                        scale = distances.std()
                        padded_type = type_.ljust(26)

                        msg += f"- {padded_type}: Mean: {loc:.2f} (+-{scale:.2f}) m\n"

                        distances = np.hstack(([0], distances))  # Exclude the value 0
                        pdf = stats.norm.pdf(distances, loc=loc, scale=scale)

                        # Create a DataFrame for the PDF data
                        pdf_data = pd.DataFrame({"distance": distances, "PDF": pdf})

                        trace = pl.plot_x_y_trace(
                            distances,
                            pdf,
                            title=f"{country}: {name} | Mean: {loc:.2f}, Std: {scale:.2f}",
                            xlabel="Distance / m",
                            ylabel="PDF",
                            color=color,
                            name=name,
                            line_property=line_property,
                        )
                        fig.append_trace(trace, row=1, col=1)

                    st.code(msg)
                    st.plotly_chart(fig)

    with tab4:
        convert = st.checkbox(
            "(Deactivated!) Convert data (rotated, shifted, neighbors)", value=False
        )
        if False and convert:
            log = st.empty()
            k = 3
            for country in st.session_state.config.countries:
                if country == "pal":
                    continue
                files = st.session_state.config.files[country]
                directory_path = Path(f"enhanced_{country}")
                if not directory_path.is_dir():
                    directory_path.mkdir()

                with st.spinner(f"Converting files for {country} ..."):
                    for selected_file in files:
                        log.info(f"{country}, {selected_file}")

                        trajectory_data = hp.load_file(selected_file)
                        data = trajectory_data.data
                        if do_rotate:
                            set_rotation_variables(selected_file, country)
                            rotated_data = hp.rotate_trajectories(
                                data,
                                st.session_state.center_x,
                                st.session_state.center_y,
                                st.session_state.angle_degrees,
                            )
                        else:
                            rotated_data = data

                        first_frame = rotated_data["frame"].to_numpy()[0]
                        # special initial conditions that make neighborhood detection difficult
                        # uppon visualisation these conditions were chosen
                        if selected_file == "ger/mix_random_4_22.csv":
                            first_frame = 600
                        if selected_file == "ger/mix_sorted_4_11.csv":
                            first_frame = 100

                        ids = rotated_data["id"].unique()
                        if len(ids) > 2:
                            nearest_dist, nearest_ind = hp.get_neighbors_at_frame(
                                first_frame, rotated_data, k
                            )
                        agents = np.unique(rotated_data["id"])
                        for agent in agents:
                            neighbors_ids = [-1, -1]
                            neighbor_types = ["prev", "next"]
                            if len(ids) > 2:
                                (
                                    neighbors,
                                    neighbors_ids,
                                    area,
                                    agent_distances,
                                    neighbor_types,
                                ) = hp.get_neighbors_special_agent_data(
                                    agent,
                                    first_frame,
                                    rotated_data,
                                    nearest_dist,
                                    nearest_ind,
                                )
                            for neighbor_id, neighbor_type in zip(
                                neighbors_ids, neighbor_types
                            ):
                                if neighbor_type == "prev":
                                    prev_neighbor = neighbor_id
                                if neighbor_type == "next":
                                    next_neighbor = neighbor_id

                            rotated_data.loc[rotated_data["id"] == agent, "prev"] = (
                                prev_neighbor
                            )
                            rotated_data.loc[rotated_data["id"] == agent, "next"] = (
                                next_neighbor
                            )
                        newfile = f"enhanced_{selected_file}"
                        log.warning(newfile)
                        rename_mapping = {
                            "id": "ID",
                            "time": "t(s)",
                            "x": "x(m)",
                            "y": "y(m)",
                        }

                        rotated_data.rename(columns=rename_mapping, inplace=True)
                        selected_columns = [
                            "ID",
                            "next",
                            "prev",
                            "gender",
                            "frame",
                            "t(s)",
                            "x(m)",
                            "y(m)",
                        ]
                        rotated_data[selected_columns].to_csv(newfile, index=False)
