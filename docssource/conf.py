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
from sys import path, argv

# Remove the source directory from path lookup to prevent aliasing
repository = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for dir in path:
    if dir.lower() == repository:
        path.remove(dir)
        
# -- Project information -----------------------------------------------------

project = 'HermesPy'
copyright = '2022, Barkhausen Institut gGmbH'
author = 'Barkhausen Institut gGmbH'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_carousel.carousel',                 # Image slideshows
    'nbsphinx',                                 # Integrate jupyter notebooks
    'sphinxcontrib.mermaid',                    # Smooth flowcahrts
    'sphinxcontrib.bibtex',                     # Latex bibliography support
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',                        # Copy script examples directly
    'sphinx_autodoc_typehints',                 # Type hinting support for the autodoc extension
    'sphinx_rtd_dark_mode',                     # Dark theme
    'sphinx_tabs.tabs',                         # Multiple tabs
    'matplotlib.sphinxext.plot_directive',      # Directly rendering plots as images
    'sphinx.ext.mathjax',                       # Rendering math equations for nbsphinx
]

autoclass_content = "both"
add_module_names = False

# Bibtex
bibtex_bibfiles = ['references.bib']
bibtex_foot_reference_style = 'foot'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffic = {
    '.rst': 'restructuredtext'
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'exclude-members': '__weakref__'
}

nbsphinx_requirejs_path = ""
nbsphinx_prolog = """

.. note::

   This static document was automatically created from the output of a jupyter notebook.
   
   Execute and modify the notebook online `here <https://colab.research.google.com/github/Barkhausen-Institut/hermespy/blob/main/docssource/{{ env.docname }}.ipynb>`_.
"""


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'images/bi.svg'
html_title = 'HermesPy Documentation'

# Sphinx RTD dark mode
default_dark_mode = True

# Carousel config
carousel_bootstrap_add_css_js = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

def setup(app):
    """Setup."""

    # Custom css tweaks
    app.add_css_file('tweaks.css')
