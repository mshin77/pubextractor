project = "pubextractor"
author = "Mikyung Shin"
copyright = "2026 Mikyung Shin"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [".doctrees"]
suppress_warnings = ["myst.header"]

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"

html_theme_options = {
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mshin77/pubextractor",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_prev_next": False,
    "show_toc_level": 1,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "secondary_sidebar_items": [],
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}

html_show_sourcelink = False

# sphinx copies every source file into _sources; the sources already live in docs_src
html_copy_source = False
