"""Show general results, including ploting, animation, ..."""

import time
from typing import TypeAlias

import pedpy
import streamlit as st
from shapely import Polygon, difference

import src.utils.helper as hp
import src.visualization.plots as pl
from src.visualization.anim import animate

st_column: TypeAlias = st.delta_generator.DeltaGenerator


def run_tab1(msg: st_column, country: str, selected_file: str) -> None:
    """First tab. Plot original data, animatoin, neighborhood."""
    c1, c2 = st.columns((1, 1))
    exterior, interior, middle_path = hp.generate_parcour()
    walkable_area = pedpy.WalkableArea(difference(Polygon(exterior), Polygon(interior)))
    msg.write("")
    if country:
        if selected_file:
            trajectory_data = hp.load_file(selected_file, sep=",")
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
            do_plot_trajectories = rc1.checkbox("Plot", value=False, help="Plot trajectories")

            do_animate = rc2.checkbox("Animation", value=False, help="Visualise movement of trajecories")
            get_neighborhood = rc3.checkbox("Neighbors", value=True, help="Calculate and visualize neighbors")

            ids = data["id"].unique()
            if do_plot_trajectories:
                c1, c2, c3 = st.columns((1, 1, 1))
                plot_parcour = c1.checkbox("Parcour", value=True)
                framerate = int(c2.slider("Every nth frame", 1, 100, 40, 10))

                uid = c3.number_input(
                    "Insert id of pedestrian",
                    value=None,
                    min_value=int(min(ids)),
                    max_value=int(max(ids)),
                    placeholder=f"Type a number in [{int(min(ids))}, {int(max(ids))}]",
                    format="%d",
                )
                # st.info(figname)
                fig = pl.plot_trajectories(data, framerate, uid, exterior, interior, plot_parcour)

                # this is for the paper to have print-quality
                # figname = f"{selected_file.split('.csv')[0]}.pdf"
                # fig_plt = pl.plot_trajectories_matplotlib(data, framerate, exterior, interior, figname=figname, plot_parcour=plot_parcour)
                # st.pyplot(fig_plt)
                st.plotly_chart(fig)
            # neighborhood
            if get_neighborhood and len(ids) > 2:
                with st.expander("How to calculate the distance? (click to expand)"):
                    st.write(
                        """
                        First, we define an oval (black line) with the following specifications:
                        - Two semi-circles with radius $r=1.65$ m.
                        - Two linear segments a length $l=2$ m.
                        - Resolution of the linear segments is $\\delta x = 0.05$ m
                        - Resolution of the angular setup is  $\\delta \\phi = \\frac{0.005}{r}$ m

                        With this we can calculate the distance between two points $p_1$ and $p_2$ with two methods:
                        - **Method 1 (Euklidean)**: Just calculate the direct distance between two points. $d_1=|p_1-p_2|$
                        - **Method 2 (Arc)**: Project the points on the oval, getting $p_1^\\prime$ and $p_2^\\prime$.
                        Then calculate the distance as $d_2 =  \\sum_i |p_i - p_{i+1} |,$ where $p_i$ are points on the arc between $p_1^\\prime$ and $p_2^\\prime$.

                        **Note**: in the graph $p_1^\\prime$ and $p_2^\\prime$ are depicted as circles.
                        """
                    )
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
                trajectory_data = pedpy.TrajectoryData(data, trajectory_data.frame_rate)
                data_with_speed = pedpy.compute_individual_speed(
                    traj_data=trajectory_data,
                    frame_step=5,
                    speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
                )
                data_with_speed = data_with_speed.merge(
                    trajectory_data.data,
                    on=["id", "frame"],
                    how="left",
                )
                color_mode = str(st.radio("Color mode", ["Speed", "Gender"]))
                every_nth_frame = int(
                    st.number_input(
                        "Every nth frame",
                        value=50,
                        min_value=1,
                        max_value=100,
                        placeholder="Every nth frame",
                        format="%d",
                    )
                )
                anm = animate(
                    data_with_speed,
                    walkable_area,
                    color_mode=color_mode,
                    width=500,
                    height=500,
                    every_nth_frame=every_nth_frame,
                    radius=0.1,  # 0.75
                    title_note="(<span style='color:green;'>M</span>, <span style='color:blue;'>F</span>)",
                )
                st.plotly_chart(anm)
