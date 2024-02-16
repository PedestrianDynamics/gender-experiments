"""This module is depcretated.

It is still existing to document how we corrected the data.
Data correction included, rotating, shifting and adding neighbors to the csv files.

This process was done once to prepare the data, so no need to run it again.

Assume: that there are directories called enhanced_{country}
"""

import streamlit as st
from pathlib import Path
import helper as hp
import numpy as np


def run_tab4(do_rotate):
    convert = st.toggle("Not active (depricated)", key="tab4", value=False)
    if False and convert:
        log = st.empty()
        k = 3
        for country in st.session_state.config.countries:
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
                        hp.set_rotation_variables(selected_file, country)
                        rotated_data = hp.rotate_trajectories(
                            data,
                            st.session_state.center_x,
                            st.session_state.center_y,
                            st.session_state.angle_degrees,
                        )
                    else:
                        rotated_data = data

                    first_frame = rotated_data["frame"].to_numpy()[0]
                    # special initial conditions that make neighborhood
                    # detection difficult
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
