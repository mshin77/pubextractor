<img src="https://mshin77.github.io/pubextractor/_static/logo.svg" alt="pubextractor Logo" width="240"/>

[![PyPI version](https://img.shields.io/pypi/v/pubextractor?cacheSeconds=0)](https://pypi.org/project/pubextractor/)
[![Python versions](https://img.shields.io/pypi/pyversions/pubextractor?cacheSeconds=0)](https://pypi.org/project/pubextractor/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Data extraction for systematic reviews and meta-analyses. Reads PDF, Word, and image files with [pdfplumber](https://github.com/jsvine/pdfplumber) and [python-docx](https://python-docx.readthedocs.io/), keeps the page, section, and table cell of every value, and codes them against an Excel codebook in an interactive [Shiny](https://shiny.posit.co/py/) interface. Exports effect sizes for group, single-case, and correlation design studies. Companion to [pubmlp](https://github.com/mshin77/pubmlp), which handles screening.

## Installation

```bash
pip install pubextractor
```

From GitHub:

```bash
pip install git+https://github.com/mshin77/pubextractor.git
```

## Load the pubextractor Package

```python
import pubextractor
```

## Alternatively, Launch and Browse the Shiny App

Access the web app (demo, short papers) at <https://www.pubextractor.org>.

Launch and browse the app on the local computer:

```python
from pubextractor import run_app

run_app()
```

Or in a terminal: `pubextractor`. The app opens at http://127.0.0.1:8000; use `run_app(port=8001)` if that port is in use.

## Getting Started

See [Quick Start](https://mshin77.github.io/pubextractor/getting-started.html) and [Using the App](https://mshin77.github.io/pubextractor/vignettes/using-the-app.html) for tutorials.

## Citation

- Shin, M. (2026). *pubextractor: Data extraction for systematic reviews and meta-analyses* (Python package version 0.1.3) [Computer software]. <https://github.com/mshin77/pubextractor>

- Shin, M. (2026). *pubextractor: Data extraction for systematic reviews and meta-analyses* [Web application]. <https://www.pubextractor.org>
