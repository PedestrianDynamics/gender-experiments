"""Main entry point to the app."""

import glob
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import streamlit as st

import enhance_data
import fundamental_diagram
import helper as hp
import proximity_analysis
import show_data
import ui


@dataclass
class DataConfig:
    """Datastructure for the app."""

    rename_mapping: Dict[str, str]
    column_types: Dict[str, str]
    countries: List[str]
    proximity_results: Dict[str, Path] = field(default_factory=dict)

    files: Dict[str, List[str]] = field(default_factory=dict)
    data: Dict[str, List] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        """Initialize the DataConfig instance by retrieving files for each country."""
        self.proximity_results["path"] = Path("app_data/proximity_analysis_results.csv")
        self.proximity_results["url"] = "https://go.fzj.de/proximity_results"
        # Ensure the directory exists
        self.proximity_results["path"].parent.mkdir(parents=True, exist_ok=True)
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
    """Init session_state. throughout the app."""
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


if __name__ == "__main__":
    # cyclomatic complexity too high (16)
    ui.ini_app_looks()
    msg = st.empty()
    init_session_state(msg)
    country, tab1, tab2, tab3, tab4 = ui.init_sidebar()
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

    with tab1:
        do_rotate = show_data.run_tab1(msg, country, selected_file)

    with tab2:
        fundamental_diagram.run_tab2(country, selected_file)

    with tab3:
        proximity_analysis.run_tab3(selected_file)

    with tab4:
        enhance_data.run_tab4(do_rotate)
