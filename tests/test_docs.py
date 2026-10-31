import re
from pathlib import Path

import pubextractor

API = Path(__file__).parents[1] / 'docs_src' / 'api.md'
SITE = Path(__file__).parents[1] / 'docs'
PAGES = ['index.html', 'getting-started.html', 'api.html', 'changelog.html', 'support.html',
         'vignettes/using-the-app.html']


def test_docs_pages_load_assets_without_version_tags():
    pages = list(SITE.rglob('*.html'))
    assert pages
    assert [p.name for p in pages if re.search(r'\.(?:js|css)\?(?:v|digest)=', p.read_text(encoding='utf-8'))] == []


def test_docs_pages_carry_a_description_and_clean_canonical():
    for page in PAGES:
        html = (SITE / page).read_text(encoding='utf-8')
        assert re.search(r'<meta content="[^"]+" name="description" />', html), page
        clean = '' if page == 'index.html' else page
        assert f'<link rel="canonical" href="https://mshin77.github.io/pubextractor/{clean}" />' in html, page


def test_docs_headings_hold_only_their_text():
    for page in PAGES:
        html = (SITE / page).read_text(encoding='utf-8')
        assert re.findall(r'<a class="headerlink"[^>]*>[^<]+</a>', html) == [], page


def test_api_reference_lists_every_export():
    text = API.read_text(encoding='utf-8')
    assert [name for name in pubextractor.__all__ if f'`{name}`' not in text] == []


def test_api_reference_lists_tracker_loggers():
    text = API.read_text(encoding='utf-8')
    loggers = ('log_model_config', 'log_random_seed', 'log_data', 'log_threshold',
               'log_criteria', 'add_note')
    assert [name for name in loggers if f'ProvenanceTracker.{name}' not in text] == []
