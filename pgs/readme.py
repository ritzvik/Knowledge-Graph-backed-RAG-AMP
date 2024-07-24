import os
import streamlit as st


cwd = os.getcwd()
readme_file = open(cwd+"/README.md", "r")
readme_text = readme_file.read()
readme_file.close()

st.markdown(readme_text, unsafe_allow_html=True)
