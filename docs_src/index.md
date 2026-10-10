# Extract data from papers

Read PDF, Word, and image files, check the source evidence, and export coded values and effect sizes.

<div class="app-actions">
<a class="app-action primary" href="https://www.pubextractor.org">Open web app</a>
<a class="app-action" href="#run-locally">Run locally</a>
</div>

The web demo accepts short papers. Run locally for full use.

<div class="review-steps">
<div><strong>1. Add papers</strong><p>Load files and select the fields to extract.</p></div>
<div><strong>2. Review evidence</strong><p>Check values, tables, figures, and source locations.</p></div>
<div><strong>3. Export results</strong><p>Download coding sheets, effect sizes, and review records.</p></div>
</div>

See [Using the app](vignettes/using-the-app.md) for the steps.

## Run locally

```bash
pip install pubextractor
```

```python
from pubextractor import run_app

run_app()
```

## Citation

- Shin, M. (2026). *pubextractor: Data extraction for systematic reviews and meta-analyses* (Python package version 0.1.3) \[Computer software\]. <a href="https://github.com/mshin77/pubextractor">https://github.com/mshin77/pubextractor</a>
- Shin, M. (2026). *pubextractor: Data extraction for systematic reviews and meta-analyses* \[Web application\]. <a href="https://www.pubextractor.org">https://www.pubextractor.org</a>

```{toctree}
:hidden:

Quick start <getting-started>
Using the app <vignettes/using-the-app>
API <api>
News <changelog>
Support <support>
```
