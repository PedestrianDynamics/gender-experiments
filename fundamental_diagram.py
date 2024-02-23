"""Run all calculations/visualisation of tab2.

tab2: Fundamental diagram
"""

import glob
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis as al
import docs
import helper as hp
import plots as pl


def load_or_calculate_fd(country: str, dv: int, diff_const: int) -> pd.DataFrame:
    """Load density calculation from file or calculate."""
    precalculated_file = f"app_data/density_micro_{country}.pkl"
    if not Path(precalculated_file).exists():
        result = al.fundamental_diagram_micro(country, dv, diff_const)
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
        docs.density_speed_documentation()
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
        diff_const = int(
            c2.slider("diff_const", 1, 500, 5, 1, help="window steady state")
        )

        if calculations == "time_series":
            al.density_speed_time_series_micro(country, selected_file, dv, diff_const)

        if calculations == calculations == "FD":
            all_data = {}
            st.divider()
            countries = [
                country
                for country in st.session_state.config.countries
                if country != "pal"
            ]
            for country in st.session_state.config.countries:
                result = load_or_calculate_fd(country, dv, diff_const)
                all_data[country] = (
                    result["individual_density"],
                    result["speed"],
                )

            st.markdown("**Fundamental diagram**")
            selected_countries = st.multiselect(
                "Select countries to display:",
                options=countries,
                default=countries,
            )
            filtered_country_data = {
                country: all_data[country] for country in selected_countries
            }
            fig = pl.plot_fundamental_diagram_all(filtered_country_data)
            hp.show_fig(fig, html=True)
