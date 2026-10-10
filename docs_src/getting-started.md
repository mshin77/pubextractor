# Getting Started

## Installation

```bash
pip install pubextractor
```


## Launch the App

```python
from pubextractor import run_app

run_app()
```

Or in a terminal:

```bash
pubextractor              # add --port 8001 for another port
```

The app opens in the browser. See [Using the App](vignettes/using-the-app.md) for the steps; **Home** in the app explains each step.

## Use in Python

```python
from pubextractor import (read_pdf, detect_page_labels, detect_sections, read_codebook_table,
                          extract_values, to_long, write_csv)

codebook = read_codebook_table("codebook.xlsx")
pages = detect_sections(detect_page_labels(read_pdf("study.pdf")))
records = extract_values(pages, codebook, paper_id="S01")
write_csv(to_long(records), "values.csv")
```
