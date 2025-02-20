# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'libyt'
copyright = '2024, Shin-Rong Tsai, Hsi-Yu Schive, Matthew Turk'
author = 'Shin-Rong Tsai, Hsi-Yu Schive, Matthew Turk'

version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx_design",
    "breathe"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# -- Options for Breathe -----------------------------------------------------
# https://breathe.readthedocs.io/en/latest/

breathe_projects = {
    "libyt": "./doxygen/xml"
}

breathe_default_project = "libyt"

# -- Options for MyST --------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist"
]
suppress_warnings = [
    "myst.header"
]
myst_heading_anchors = 6

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -- Read the Doc Hook -------------------------------------------------------
if os.environ.get("READTHEDOCS") == "True":
    def generate_doxygen_xml(_):
        subprocess.call('cd doc; doxygen', shell=True)


    def setup(app):
        app.connect('builder-inited', generate_doxygen_xml)
