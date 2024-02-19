""" Run logic for tab3.

tab3: proximity analysis.
"""

import subprocess
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import helper as hp
import plots as pl


def run_tab3(selected_file: str):
    tab3_on = st.toggle("Activate", value=False)
    if tab3_on:
        do_analysis = st.radio(
            "Choose option",
            [
                "load_gender_analysis_(Euklidean)",
                "load_gender_analysis_(Arc)",
                # "calculate_gender_analysis",
                # "plot_existing_data",
            ],
            horizontal=True,
        )

        if do_analysis == "calculate_gender_analysis":
            start_time = time.time()

            with st.status("Calculating ...", expanded=True):
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

        if (
            do_analysis == "load_gender_analysis_(Euklidean)"
            or do_analysis == "load_gender_analysis_(Arc)"
        ):
            if do_analysis == "load_gender_analysis_(Euklidean)":
                result_csv = st.session_state.config.proximity_results0["path"]
                csv_url = st.session_state.config.proximity_results0["url"]
            else:
                result_csv = st.session_state.config.proximity_results["path"]
                csv_url = st.session_state.config.proximity_results["url"]

            msg = st.empty()
            st.divider()
            c0, c1, c2, c3, c4, c5 = st.columns(6)
            st.divider()

            if not result_csv.exists():
                msg.warning(f"{result_csv} does not exist yet!")
                with st.status("Downloading ...", expanded=True):
                    hp.download_csv(csv_url, result_csv)

            if result_csv.exists():
                msg.info(f"Reading file {result_csv}")
                proximity_df = pd.read_csv(result_csv)
            else:
                msg.warning(f"File {result_csv} not found.")
                st.stop()

            # maybe filter low densities out. File with only 4 pedestrians
            # exclude = c5.checkbox(
            #     "Exclude *04-files",
            #     value=True,
            #     help="Exclude 04 files, since their neighborhood may not be correct.",
            # )
            # if exclude:
            #     pattern = ".*_04_.*\.csv"
            #     proximity_df = proximity_df[~proximity_df["file"].str.contains(pattern)]
            msg.empty()
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
            show_dataframe = c0.checkbox("Show data", value=False)
            ttest = c1.checkbox(
                "T-test", value=False, help="Perform ttest on the distances."
            )
            plot_pdf = c2.checkbox(
                "PDF", value=False, help="PDF function for distances"
            )
            debug = c3.checkbox(
                "Time-series",
                value=False,
                help="Plot times series of distance per file",
            )
            plot_box = c4.checkbox(
                "Box-plot",
                value=False,
                help="Plot box plot for all countries **(slow!)**.",
            )
            if show_dataframe:
                col1, col2, col3 = st.columns((0.3, 0.3, 0.3))
                page_size = col3.number_input(
                    "Number of rows",
                    value=100,
                    min_value=100,
                    max_value=int(len(proximity_df) / 2),
                )

                with col1:
                    decrement = st.button("Previous Page")
                    if decrement:
                        hp.decrement_page_start(page_size)
                with col2:
                    increment = st.button("Next Page")
                    if increment:
                        hp.increment_page_start(page_size)

                if st.session_state.page_start < 0:
                    st.session_state.page_start = 0

                # Ensure page_start doesn't go above total data length
                if st.session_state.page_start >= len(proximity_df):
                    st.session_state.page_start = len(proximity_df) - page_size

                # Calculate page_end
                page_end = st.session_state.page_start + page_size
                st.dataframe(proximity_df.iloc[st.session_state.page_start : page_end])

            if debug:
                # st.info(f"Data for {selected_file}")
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
                hp.show_fig(fig, html=True)
                st.dataframe(proximity_df.loc[proximity_df["file"] == selected_file])
            if ttest:
                type_ = st.radio(
                    "Select group",
                    ["female", "male", "mix_random", "mix_sorted", "all"],
                    horizontal=True,
                )
                st.markdown(
                    ":information_source: **The mean distance between pairs of the same gender is equal to the mean distance between pairs of different genders.**"
                )
                st.latex("p <= 0.05 \\rightarrow \\text{ reject}\; H_0")

                for country in ["aus", "ger", "jap", "chn"]:
                    msg = ""
                    filtered_data = proximity_df[
                        (proximity_df["country"] == country)
                        & (proximity_df["type"] == type_)
                    ]
                    if type_ == "all":
                        # If "all" is selected, filter by country only
                        filtered_data = proximity_df[proximity_df["country"] == country]
                    else:
                        # Otherwise, filter by both country and type
                        filtered_data = proximity_df[
                            (proximity_df["country"] == country)
                            & (proximity_df["type"] == type_)
                        ]
                    # st.dataframe(filtered_data)
                    with st.status(
                        f"Calculating T-tests for **<{country}>** and **<{type_}>**",
                        expanded=True,
                    ):
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

                        msg += f"- Next neighbors, different-same: T-Statistic = {t_stat_next:.02f}, P-Value = {p_val_next:.02f}\n"
                        msg += f"- Prev neighbors, different-same: T-Statistic = {t_stat_prev:.02f}, P-Value = {p_val_prev:.02f}"
                        st.info(msg)

            if plot_pdf:
                proximity_melted["distance"] = proximity_melted["distance"].fillna(0)
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
                        # pdf_data = pd.DataFrame({"distance": distances, "PDF": pdf})

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

            if plot_box:
                with st.status("Plotting ..."):
                    fig = px.box(
                        proximity_melted,
                        x="category",
                        y="distance",
                        color="country",
                        title="Proximity Analysis Based on Gender and Country",
                        labels={
                            "distance": "Proximity Distance",
                            "category": "Category",
                        },
                    )

                    fig.update_layout(
                        yaxis_title="Distance",
                        xaxis_title="Gender Proximity Category",
                        showlegend=True,
                    )

                    hp.show_fig(fig)
