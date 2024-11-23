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
    current_file_path = Path(__file__)
    ROOT_DIR = current_file_path.parent.parent.absolute()
    logo_path = ROOT_DIR / ".." / "images" / "logo.png"
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    zenodo_badge = "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12675716.svg)](https://doi.org/10.5281/zenodo.12675716)"
    data_badge = "[![DOI:10.34735/ped.2024.1](https://img.shields.io/badge/DOI-10.34735/ped.2024.1-blue.svg)](https://doi.org/10.34735/ped.2024.1)"
    article_badge = "[![DOI:10.1016/j.ssci.2024.106710](https://img.shields.io/badge/Safety%20Science-Published-blue.svg)](https://doi.org/10.1016/j.ssci.2024.106710)"
    repo = "https://github.com/PedestrianDynamics/gender-experiments"
    repo_name = f"[![Repo]({gh})]({repo})"
    c1, c2 = st.sidebar.columns((0.25, 0.8))
    c1.write("**Code**")
    c2.write(zenodo_badge)
    c1.write("**Data**")
    c2.write(data_badge)
    c1.write("**Repo**")
    c2.markdown(repo_name, unsafe_allow_html=True)
    c1.write("**Article**")
    c2.markdown(article_badge, unsafe_allow_html=True)
    st.sidebar.image(str(logo_path), use_column_width=True)
    # use_column_width will be deprecated and replaced with use_contrainer_widthd
    # st.sidebar.image(str(logo_path), use_container_width=True)


def init_sidebar() -> Tuple[str, DeltaGenerator, DeltaGenerator, DeltaGenerator, DeltaGenerator]:
    """Init sidebar and tabs."""
    c1, c2 = st.sidebar.columns((1.8, 0.2))
    flag = c2.empty()
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👫🏻 View trajectories",
            "📉 Fundamental diagram",
            "📍 Proximity analysis",
            "📂 Pair distribution function",
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
