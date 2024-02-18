""" Show general results, including ploting, animation, ..."""

import time

import pedpy
import streamlit as st
from shapely import Polygon, difference

import helper as hp
import plots as pl
from anim import animate
from pathlib import Path
import numpy as np
import pandas as pd


def run_tab1(msg, country: str, selected_file: str):
    """First tab. Plot original data, animatoin, neighborhood."""
    c1, c2 = st.columns((1, 1))
    do_rotate = False
    exterior, interior, middle_path = hp.generate_parcour()
    walkable_area = pedpy.WalkableArea(difference(Polygon(exterior), Polygon(interior)))
    msg.write("")
    if country:
        if selected_file:

            # st.toast(f"{selected_file}", icon="📂")
            # file_index = files.index(selected_file)
            # default values
            hp.set_rotation_variables(selected_file, country)
            # trajectory_data = st.session_state.loaded_data[country][file_index]
            trajectory_data = hp.load_file(selected_file)
            data = trajectory_data.data
            if selected_file != st.session_state.file_changed:
                st.session_state.file_changed = selected_file
                st.session_state.new_data = data.copy()

            # st.dataframe(data)
            start_time = time.time()
            rc0, rc1, rc2, rc3 = st.columns((1, 1, 1, 1))
            st.write("---------")
            columns_to_display = ["id", "frame", "time", "x", "y", "prev", "next"]
            display = rc0.checkbox("Data", value=False, help="Display data table")
            if display:
                st.dataframe(trajectory_data.data.loc[:, columns_to_display])
            do_plot_trajectories = rc1.checkbox(
                "Plot", value=False, help="Plot trajectories"
            )

            do_animate = rc2.checkbox(
                "Animation", value=False, help="Visualise movement of trajecories"
            )
            get_neighborhood = rc3.checkbox(
                "Neighbors", value=True, help="Calculate and visualize neighbors"
            )
            ids = data["id"].unique()
            if do_plot_trajectories:
                # do_fix = c1.checkbox("Fix", value=False)
                # if do_fix:
                #     hp.set_rotation_variables(selected_file, country)
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
            if get_neighborhood and len(ids) > 2:
                # if selected_file == "ger/mix_random_4_22.csv":
                #     first_frame = 600
                #     data = data[data["frame"] >= first_frame]
                # if selected_file == "ger/mix_sorted_4_11.csv":
                #     first_frame = 100
                #     data = data[data["frame"] >= first_frame]
                # ===============================================
                if country == "pal0":
                    for filename in st.session_state.config.files[country]:
                        trajectory_data = hp.load_file(filename)
                        data = trajectory_data.data
                        data[["next", "prev"]] = np.nan
                        st.info(f"init neighbors for {filename}")
                        frames = data["frame"].to_numpy()
                        for fr in frames:
                            data0 = data[data["frame"] == fr].copy()
                            data0_sorted = data0.sort_values(by="x")
                            data0_sorted.reset_index(drop=True, inplace=True)
                            sorted_ids = data0_sorted["id"].tolist()
                            for index, current_id in enumerate(sorted_ids):
                                prev_id = sorted_ids[index - 1] if index > 0 else None
                                next_id = (
                                    sorted_ids[index + 1]
                                    if index < len(sorted_ids) - 1
                                    else None
                                )
                                # Fetch the current 'prev' and 'next' values to check if they are NaN
                                current_prev = data.loc[
                                    data["id"] == current_id, "prev"
                                ].values
                                current_next = data.loc[
                                    data["id"] == current_id, "next"
                                ].values
                                # Update 'prev' if it's NaN
                                if len(current_prev) > 0 and pd.isna(current_prev[0]):
                                    data.loc[data["id"] == current_id, "prev"] = prev_id

                                # Update 'next' if it's NaN
                                if len(current_next) > 0 and pd.isna(current_next[0]):
                                    data.loc[data["id"] == current_id, "next"] = next_id
                        # write to file
                        directory = Path("enhanced_" + filename.split("/")[0])
                        directory.mkdir(parents=True, exist_ok=True)
                        newfile = f"enhanced_{filename}"
                        st.warning(newfile)
                        rename_mapping = {
                            "id": "ID",
                            "time": "t(s)",
                            "x": "x(m)",
                            "y": "y(m)",
                        }
                        wdata = data.copy()
                        wdata.rename(columns=rename_mapping, inplace=True)
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

                        wdata[selected_columns].to_csv(newfile, index=False)

                    # ===============================================
                fig, new_data = hp.plot_neighbors_analysis(
                    selected_file,
                    st.session_state.new_data,
                    ids,
                    exterior,
                    interior,
                    middle_path,
                )
                st.session_state.new_data = new_data
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
