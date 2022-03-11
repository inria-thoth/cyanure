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
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_cyanure(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no--build-isolation", package])

install("numpy")
install("scipy")
install("scikit-learn")
install_cyanure("../../../")
sys.path.insert(0, os.path.abspath('../../../'))


# -- Project information -----------------------------------------------------

project = 'Cyanure'
copyright = '2019, Julien Mairal'
author = 'Julien Mairal'

# The full version, including alpha/beta/rc tags
release = '0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.mathjax',
        'sphinx.ext.viewcode',
        'sphinx.ext.autosummary',
        'sphinx.ext.doctest',
        'sphinx.ext.inheritance_diagram',
        'numpydoc',
        'jupyter_sphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
autosummary_generate = True
numpydoc_show_class_members = False 

source_encoding = 'utf-8'
#source_encoding = 'iso-8859-15'
language = 'en'
html_output_encoding = 'utf-8'
#html_output_encoding = 'ISO-8859-1'
