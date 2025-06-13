"""Sphinx configuration file for torch-secorder documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Torch-Secorder"
copyright = "2025, PyBrainn"
author = "PyBrainn"

# The full version, including alpha/beta/rc tags
release = "0.0.1"
version = release  # The short X.Y version

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx.ext.autosummary",
    "myst_parser",
]

html_logo = "_static/logo.png"

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use for HTML and HTML Help pages
html_theme = "sphinx_book_theme"
html_title = f"{project} v{release}"

# Theme options
html_theme_options = {
    "repository_url": "https://github.com/pybrainn/torch-secorder",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "navigation_depth": 2,
    "logo": {"text": "Torch-Secorder"},
}

# Add any paths that contain custom static files
html_static_path = ["_static"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST Parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}
