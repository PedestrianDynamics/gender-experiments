"""Data structure, loading files and initialising session_state."""

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


@dataclass
class DataConfig:
    """Datastructure for the app."""

    rename_mapping: Dict[str, str]
    column_types: Dict[str, Any]  # actually its a type
    countries: List[str]
    # Arc distance
    proximity_results_arc: Dict[str, Any] = field(default_factory=dict)
    # Euklidean distance
    proximity_results_euc: Dict[str, Any] = field(default_factory=dict)

    files: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the DataConfig instance by retrieving files for each country."""
        # direct distance results
        # Euc
        self.proximity_results_euc["path"] = Path("app_data/proximity_analysis_results_euc.csv")
        self.proximity_results_euc["url"] = "https://go.fzj.de/proximity_results_euc"
        # Arc
        self.proximity_results_arc["path"] = Path("app_data/proximity_analysis_results_arc.csv")
        self.proximity_results_arc["url"] = "https://go.fzj.de/proximity_results_arc"

        # Ensure the directory exists
        self.proximity_results_euc["path"].parent.mkdir(parents=True, exist_ok=True)
        self.proximity_results_arc["path"].parent.mkdir(parents=True, exist_ok=True)
        self.retrieve_files()

    def retrieve_files(self) -> None:
        """Retrieve the files for each country specified in the countries list.

        The files are expected to be CSV files located in directories named after the countries.

        This method updates the `files` dictionary with country names as keys and lists
        of file paths as values.
        """
        for country in self.countries:
            self.files[country] = glob.glob(f"data/{country}/*.csv")


def init_session_state(msg: DeltaGenerator) -> None:
    """Init session_state. throughout the app."""
    if "file_changed" not in st.session_state:
        st.session_state.file_changed = ""

    if "new_data" not in st.session_state:
        st.session_state.new_data = pd.DataFrame()
    if "page_start" not in st.session_state:
        st.session_state.page_start = 0

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
