# Using the App

## Start

```bash
pip install pubextractor
```

```python
from pubextractor import run_app

run_app()  # opens http://127.0.0.1:8000
```

Or run `pubextractor` in a terminal.

From a source checkout, `python -m shiny run pubextractor/app.py` starts the same app.

## Steps

1. **Papers**: pick Upload Example Files, or Upload Your File for PDF, DOCX, or image files. Extraction starts once papers are added.
2. **Codebook**: import the Excel sheet or edit the table in the app. Extraction runs again after each change and keeps coder edits.
3. **Model**: rules only by default. A local model or an API model is optional.
4. **Review**: check Values, Tables, Trace, and Figures.
5. **Export**: download the CSV files. Compare with a second coder's sheet for agreement.

## Try it

The package ships a codebook (`codebook.xlsx`, `codebook.yml`), a PDF (`study-s01.pdf`), and a DOCX (`study-s02.docx`) in `pubextractor/examples/`. Upload Example Files in Papers loads both papers; the default codebook is `codebook.yml`.

## Help

Details for each step are under **Home** in the app.
