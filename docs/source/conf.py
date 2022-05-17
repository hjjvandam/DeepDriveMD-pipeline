# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))
import deepdrivemd  # noqa


# -- Project information -----------------------------------------------------

project = "deepdrivemd"
author = "Alexander Brace, Hyungro Lee, Heng Ma, Anda Trifan, Matteo Turilli, Igor Yakushin, Li Tan, Andre Merzky, Tod Munson, Ian Foster, Shantenu Jha, Arvind Ramanathan"
now = datetime.datetime.now()
copyright = "2020-{}, ".format(now.year) + author

# The full version, including alpha/beta/rc tags
release = deepdrivemd.__version__
version = deepdrivemd.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# List of imports to mock when building the documentation.
autodoc_mock_imports = [
    "adios2",
    "tensorflow",
    "simtk.openmm",
    "cupy",
    "cuml",
    "numba",
    "torch",
    "torchsummary",
    "mdlearn",
]

html_context = {
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# Include __init__()
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
