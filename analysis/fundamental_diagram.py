"""Run all calculations/visualisation of tab2.

tab2: Fundamental diagram
"""

import glob
import os
import pickle
import time
from pathlib import Path

import pandas as pd
import pedpy as pp
import plotly.graph_objects as go
import streamlit as st
from typing import List

import utils.helper as hp
import visualization.plots as pl
from analysis.measurement import (  # calculate_speed,
    calculate_individual_density_csv,
    calculate_steady_state,
    density_speed_time_series_micro,
)
from utils.docs import density_speed_documentation


def fundamental_diagram_all_countries(method: str, df: pd.DataFrame, dv: int, diff_const: int) -> go.Figure:
    """Calculate the functamental diagram with a specific dist calculation.

    df contains the content of the csv file with the distance calculations.
    """
    all_data = {}
    for country in st.session_state.config.countries:
        result = load_or_calculate_fd(method, df, country, dv, diff_const)
        all_data[country] = (
            result["individual_density"],
            result["speed"],
        )

    st.markdown("**Fundamental diagram**")
    selected_countries: List[str] = st.multiselect(
        "Select countries to display:",
        key=method,
        options=st.session_state.config.countries,
        default=st.session_state.config.countries,
    )
    filtered_country_data = {country: all_data[country] for country in selected_countries}
    return pl.plot_fundamental_diagram_all(filtered_country_data)


def fundamental_diagram_micro(df: pd.DataFrame, country: str, dv: int, diff_const: int) -> pd.DataFrame:
    """Calculate FD from results csv file."""
    all_merged_df = pd.DataFrame()
    msg = st.empty()
    c1, c2 = st.columns((1, 1))
    with msg.status(f"Calculating {country} ...", expanded=False):
        start_time = time.time()
        for filename in st.session_state.config.files[country]:
            try:
                new_path = "/".join(Path(filename).parts[1:])
                trajectory_data = hp.load_file(filename)
                # data = trajectory_data.data
                filter_df = df[(df["country"] == country) & (df["file"] == new_path)]
                density = calculate_individual_density_csv(filter_df)
                # speed = calculate_speed(data, dv)
                speed = pp.compute_individual_speed(
                    traj_data=trajectory_data,
                    frame_step=dv,
                    speed_calculation=pp.SpeedCalculation.BORDER_SINGLE_SIDED,
                )

                steady_state_index = calculate_steady_state(speed, window_size=5, threshold=0.1, diff_const=diff_const)
                speed_df = speed.loc[:, ["frame", "id", "speed"]].iloc[steady_state_index:]

                # Data consistency check (example)

                if not density.empty and not speed_df.empty:
                    merged_df = pd.merge(density, speed_df, on=["id", "frame"])
                    all_merged_df = pd.concat([all_merged_df, merged_df], ignore_index=True)
                else:
                    msg.warning(f"Empty DataFrame encountered for {filename}. Skipping merge.")
            except Exception as e:
                msg.error(f"Error processing {filename}: {e}")

    end_time = time.time()
    msg.info(f"Finished with {filename} in {end_time-start_time:.2f} s")
    msg.empty()
    return all_merged_df


def load_or_calculate_fd(method: str, df: pd.DataFrame, country: str, dv: int, diff_const: int) -> pd.DataFrame:
    """Load density calculation from file or calculate."""
    precalculated_file = f"app_data/density_micro_{method}_{country}.pkl"
    if not Path(precalculated_file).exists():
        result = fundamental_diagram_micro(df, country, dv, diff_const)
        with open(precalculated_file, "wb") as f:
            pickle.dump(result, f)
    else:
        print(f"load precalculated file {precalculated_file}")
        with open(precalculated_file, "rb") as f:
            result = pickle.load(f)

    if result.empty:
        st.error("Something went south.")
        st.stop()

    return result


def run_tab2(country: str, selected_file: str) -> None:
    """Contain main logic of tab2: FD diagram."""
    do_calculations = st.toggle("Activate", key="tab2", value=False)
    docs_expander = st.expander("Documentation (click to expand)", expanded=False)
    with docs_expander:
        density_speed_documentation()
    c0, c1, c2 = st.columns((1, 1, 1))
    if do_calculations:
        c2.write("**Speed calculation parameters**")
        calculations = c0.radio(
            "Choose calculation",
            [
                # "micro_fd_rudina",
                "time_series",
                "FD",
            ],
        )
        if c1.button(
            "Delete files",
            help="To improve efficiency, certain density and speed values are pre-loaded rather than dynamically computed. By using this button, you have the option to remove these pre-loaded files, allowing for fresh calculations to be initiated from the beginning.",
        ):
            precalculated_files_pattern = "app_data/*.pkl"
            files_to_delete = glob.glob(precalculated_files_pattern)
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    st.toast(f"Deleted {file_path}", icon="✅")
                except Exception as e:
                    st.error(f"Error deleting {file_path}: {e}")

        dv = int(
            c2.slider(
                r"$\Delta t$",
                1,
                100,
                10,
                5,
                help="To calculate the displacement over a specified number of frames. See Eq. (1)",
            )
        )
        diff_const = int(c2.slider("diff_const", 1, 500, 5, 1, help="window steady state"))

        if calculations == "time_series":
            density_speed_time_series_micro(country, selected_file, dv, diff_const)

        if calculations == calculations == "FD":
            st.divider()
            paths = [
                st.session_state.config.proximity_results_euc["path"],
                st.session_state.config.proximity_results_arc["path"],
            ]
            urls = [
                st.session_state.config.proximity_results_euc["url"],
                st.session_state.config.proximity_results_arc["url"],
            ]
            methods = ["Euklidean", "Arc"]
            for i, (result_csv, url) in enumerate(zip(paths, urls)):
                if not result_csv.exists():
                    st.warning(f"{result_csv} does not exist yet!")
                    with st.status("Downloading ...", expanded=True):
                        hp.download_csv(url, result_csv)

                if result_csv.exists():
                    st.info(f"Reading file {result_csv}")
                    df = pd.read_csv(result_csv)

                fig = fundamental_diagram_all_countries(methods[i], df, dv, diff_const)
                hp.show_fig(fig, html=True)
