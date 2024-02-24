"""Main entry point to the app."""

import streamlit as st
from pathlib import Path
import fundamental_diagram
import helper as hp
import proximity_analysis
import show_data
import ui
import datafactory

if __name__ == "__main__":
    ui.init_page_config()
    ui.init_app_looks()
    msg = st.empty()
    datafactory.init_session_state(msg)
    country, tab1, tab2, tab3, tab4 = ui.init_sidebar()
    files = st.session_state.config.files[country]
    n_female, n_male, n_mixed_random, n_mixed_sorted = hp.get_numbers_country(country)
    st.sidebar.info(
        f" Number files: {len(files)}\n- Female files: {n_female}\n- Male files: {n_male}\n- Mix sorted files: {n_mixed_sorted}\n- Mix random files: {n_mixed_random}"
    )
    file_names = [f.split("/")[-1] for f in files]
    sorted_file_names = sorted(file_names, key=hp.sorting_key)
    selected_file = str(
        st.sidebar.radio("Select a file", sorted_file_names, horizontal=True)
    )
    data_dir = Path("data")
    country_dir = data_dir / country
    selected_file = country_dir / selected_file

    with tab1:
        show_data.run_tab1(msg, country, selected_file)

    with tab2:
        fundamental_diagram.run_tab2(country, selected_file)

    with tab3:
        proximity_analysis.run_tab3(selected_file)

    with tab4:
        st.info("Will be deleted soon!")
    #     enhance_data.run_tab4(do_rotate)
