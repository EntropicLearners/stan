import os
import shutil
import sys
from datetime import datetime

import mock

# Path setup
# ----------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))


# Project information 
# -------------------

project = 'Stan'
copyright = '2025, Eric M. Furst and Vasu Venkateshwaran'
author = 'Eric M. Furst and Vasu Venkateshwaran'
release = '0.0.1.alpha'

# General configuration 
# ---------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_nb",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

add_module_names = False
autosummary_generate = True
globaltoc_maxdepth = 2

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = "index"
pygments_style = "sphinx"
todo_include_todos = False


# Options for HTML output 
# -----------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/logo.png"
html_js_files = ["custom.js"]
html_css_files = ["custom.css"]
html_static_path = ["_static"]
# If false, no module index is generated.
html_domain_indices = True
# If false, no index is generated.
html_use_index = True
# If true, the index is split into individual pages for each letter.
html_split_index = False
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

html_theme_options = {
    "external_links": [
        {"name": "Issue Tracker", "url": "https://github.com/vasudevanv/stan/issues"}
    ],
    "navigation_with_keys": False,
    "show_prev_next": False,
    "icon_links_label": "Quick Links",
    "use_edit_page_button": False,
    "navbar_align": "left",
}

html_sidebars = {}