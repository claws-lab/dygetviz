# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import doctest
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import os.path as osp
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../dygetviz'))
print(os.path.abspath('../dygetviz'))


source_suffix = '.rst'
master_doc = 'index'
doctest_default_flags = doctest.NORMALIZE_WHITESPACE
project = 'DyGETViz'
copyright = '2023, Yiqiao Jin, Andrew Zhao'
author = 'Yiqiao Jin, Andrew Zhao'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '../dygetviz/plot_dash.py', '../dygetviz/plot_dash_server.py',
                    '../dygetviz/plot_dash_server_wholeplots.py', '../dygetviz/plot_dtdg.py']

for file in exclude_patterns:
    if not osp.exists(file):
        print(f'{file} does NOT exist!')


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']

def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)


