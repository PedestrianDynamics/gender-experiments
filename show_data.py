""" Show general results, including ploting, animation, ..."""

import time

import pedpy
import streamlit as st
from shapely import Polygon, difference

import helper as hp
import plots as pl
from anim import animate


def run_tab1(msg, country: str, selected_file: str):
    """First tab. Plot original data, animatoin, neighborhood."""
    c1, c2 = st.columns((1, 1))
    do_rotate = False
    exterior, interior = hp.generate_parcour()
    walkable_area = pedpy.WalkableArea(difference(Polygon(exterior), Polygon(interior)))
    msg.write("")
    if country:
        if selected_file:
            # file_index = files.index(selected_file)
            # default values
            hp.set_rotation_variables(selected_file, country)
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
