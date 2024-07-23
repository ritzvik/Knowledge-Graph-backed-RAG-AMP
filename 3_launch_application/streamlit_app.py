import os
import streamlit as st

cwd = os.getcwd()

st.set_page_config(layout="wide")
pg = st.navigation(
    [
        st.Page(cwd+"/pgs/llm_selection.py", title="LLM Selection", icon=":material/tv_options_edit_channels:"),
        st.Page(cwd+"/pgs/rag_app.py", title="Q/A for AI/ML research papers", icon=":material/description:"),
    ]
)
pg.run()
