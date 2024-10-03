"""Main entry point to the app."""

from pathlib import Path
import streamlit as st
from src.utils.logger_config import setup_logging
from src.analysis.fundamental_diagram import run_tab2

# from analysis.proximity_analysis import run_proximity_script,
from src.analysis.proximity_analysis import run_tab3, run_pair_distribution
from src.datafactory import init_session_state
from src.utils.helper import get_numbers_country, sorting_key, download_and_extract_zip
from src.utils.ui import init_app_looks, init_page_config, init_sidebar
from src.visualization.show_data import run_tab1


if __name__ == "__main__":
    setup_logging()
    init_page_config()
    msg = st.empty()
    #############
    # Call the function to download and extract the data
    print("DOWNLIND ZIP ...")
    download_and_extract_zip()
    print("DONE with downloading ZIP")
    #############
    init_app_looks()
    init_session_state(msg)
    country, tab1, tab2, tab3, tab4 = init_sidebar()
    files = st.session_state.config.files[country]
    n_female, n_male, n_mixed_random, n_mixed_sorted = get_numbers_country(country)
    st.sidebar.info(f" Number files: {len(files)}\n- Female files: {n_female}\n- Male files: {n_male}\n- Mix sorted files: {n_mixed_sorted}\n- Mix random files: {n_mixed_random}")
    file_names = [f.split("/")[-1] for f in files]
    sorted_file_names = sorted(file_names, key=sorting_key)
    selected_file = str(st.sidebar.radio("Select a file", sorted_file_names, horizontal=True))
    data_dir = Path("data")
    country_dir = data_dir / country
    selected_file = str(country_dir / selected_file)

    with tab1:
        run_tab1(msg, country, selected_file)

    with tab2:
        run_tab2(country, selected_file)

    with tab3:
        run_tab3(selected_file)

    with tab4:
        run_pair_distribution(selected_file)
