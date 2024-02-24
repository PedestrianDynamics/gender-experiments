"""Module for UI organisation."""

from pathlib import Path
from typing import Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


def init_page_config() -> None:
    """Set up information that show on the webpage."""
    st.set_page_config(
        page_title="Single-file experiments with different gender and countries",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/PedestrianDynamics/gender-experiments",
            "Report a bug": "https://github.com/PedestrianDynamics/gender-experiments/issues",
            "About": "# Gender-experiments across culture.\n This is a tool to analyse and visualise several experiments of pedestrian dynamics in five different countries:\n\n :flag-ac: Australia, :flag-cn: China, :flag-jp: Japan, :flag-de: Germany and :flag-ps: Palestine.",
        },
    )


def init_app_looks() -> None:
    """Add badges to sidebar."""
    path = Path(__file__)
    ROOT_DIR = path.parent.absolute()

    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/PedestrianDynamics/gender-experiments"
    repo_name = f"[![Repo]({gh})]({repo})"
    c1, c2 = st.sidebar.columns((1.2, 0.5))
    c2.markdown(repo_name, unsafe_allow_html=True)
    # TODO: update till after the release
    # c1.write(
    #     "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7697604.svg)](https://doi.org/10.5281/zenodo.7697604)"
    # )
    st.sidebar.image(f"{ROOT_DIR}/logo.png", use_column_width=True)


def init_sidebar() -> (
    Tuple[str, DeltaGenerator, DeltaGenerator, DeltaGenerator, DeltaGenerator]
):
    """Init sidebar and tabs."""
    c1, c2 = st.sidebar.columns((1.8, 0.2))
    flag = c2.empty()
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👫🏻 View trajectories",
            "📉 Fundamental diagram",
            "📍 Proximity analysis",
            "📂 write enhanced data",
        ]
    )

    country = str(c1.selectbox("Select a country:", st.session_state.config.countries))
    if "jap" in country:
        flag.write(":flag-jp:")
    if "aus" in country:
        flag.write(":flag-ac:")
    if "chn" in country:
        flag.write(":flag-cn:")
    if "ger" in country:
        flag.write(":flag-de:")
    if "pal" in country:
        flag.write(":flag-ps:")

    return country, tab1, tab2, tab3, tab4
