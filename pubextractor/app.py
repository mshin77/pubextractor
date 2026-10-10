"""Shiny app for pubextractor: read papers, load a codebook, extract, review, and export.

Run with ``pubextractor.run_app()`` or ``python -m shiny run pubextractor/app.py``.
"""

import argparse
import asyncio
import base64
import hashlib
import html
import inspect
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
import zlib
from collections import Counter
from functools import cache, wraps
from importlib.resources import files
from itertools import groupby
from pathlib import Path
from urllib.parse import quote

# shiny run puts this folder first on sys.path, where statistics.py would shadow the stdlib module
_HERE = Path(__file__).resolve().parent
if __name__ != 'pubextractor.app':
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != _HERE]
    sys.path.insert(0, str(_HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import yaml  # noqa: E402
from htmltools import HTMLDependency  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402
from shiny import App, reactive, render, ui  # noqa: E402
from shiny.types import SilentException  # noqa: E402

import pubextractor as px  # noqa: E402
from pubextractor.output import COLUMNS  # noqa: E402

EXAMPLES = files('pubextractor').joinpath('examples')
DEFAULT_CODEBOOK = EXAMPLES.joinpath('codebook.yml').read_text(encoding='utf-8')
EXAMPLE_PAPERS = ('study-s01.pdf', 'study-s02.docx')
PAPER_SOURCES = {'': 'Select papers', 'example': 'Upload Example Files',
                 'upload': 'Upload Your File'}

TABLE_COLUMNS = ('field', 'value', 'status', 'arm', 'outcome', 'stat', 'table', 'row', 'col')
LONG_COLUMNS = ('paper_id', 'field', 'value', 'layer', 'status', 'pdf_page', 'printed_page',
                'section', 'quote')
EFFECT_VIEW = ('es_id', 'outcome', 'time', 'treatment', 'control', 'n_t', 'n_c', 'g', 'se_g',
               'method', 'status', 'checks')
STATUSES = ('extracted', 'review', 'not_found', 'refused')
STATUS_COLORS = {'extracted': '#198754', 'review': '#e5a100', 'not_found': '#6c757d',
                  'refused': '#dc3545'}
STATUS_BADGES = {'extracted': 'text-bg-success', 'review': 'text-bg-warning',
                 'not_found': 'text-bg-secondary', 'refused': 'text-bg-danger'}
API_DEFAULTS = {'openai': 'gpt-4.1-mini',
                'gemini': os.environ.get('GEMINI_MODEL', '').strip() or 'gemini-2.5-flash'}
LICENSES = {'': 'None stated', 'cc-by': 'CC BY', 'cc-by-sa': 'CC BY-SA', 'cc0': 'CC0',
            'public-domain': 'Public domain', 'cc-by-nc': 'CC BY-NC', 'cc-by-nd': 'CC BY-ND',
            'cc-by-nc-sa': 'CC BY-NC-SA', 'cc-by-nc-nd': 'CC BY-NC-ND'}
ENCODER_NAME = 'all-MiniLM-L6-v2'
PALETTE = ('#007bc2', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2',
           '#17becf')
CATEGORY = ('category', 'multi_category')
READABLE = ('.pdf', '.docx', *px.tables.IMAGE_FILES)
IMAGE_KINDS = {suffix.lstrip('.') for suffix in px.tables.IMAGE_FILES}
SKIP_REASON = 'only PDF, DOCX, or image'
SKIP_ERROR = 'only PDF, DOCX, or image files are read'
UPLOAD_COLUMNS = ('paper', 'file', 'size', 'status')
HOSTED = os.environ.get('PUBEXTRACTOR_HOSTED', '').strip().lower() in ('1', 'true', 'yes', 'on')
HOSTED_FILES, HOSTED_PAGES, HOSTED_BYTES = 3, 20, 10 * 1024 ** 2
COPYRIGHT_CHOICES = {'open_access': 'Open access (CC BY or similar)',
                     'personal': 'My own paper or preprint',
                     'subscription': 'Subscription paper from a journal', 'unsure': 'Not sure'}
STOP_UPLOAD = ('Subscription papers stay on the computer that holds them: install pubextractor '
               'and run it locally.')
DEMO_NOTE = ('Demo: short papers only (up to 3 files, 20 pages or 10 MB each). Upload only papers '
             'that may be shared; files are deleted when the page closes. ')
AI_LIMITS = {'session': ('PUBEXTRACTOR_AI_SESSION_CALLS', 20), 'client': ('PUBEXTRACTOR_AI_CLIENT_CALLS', 20),
             'hour': ('PUBEXTRACTOR_AI_SHARED_CALLS', 60), 'day': ('PUBEXTRACTOR_AI_SHARED_DAILY', 300)}
AI_CALLS = []
READ_COLUMNS = ('paper', 'file', 'size', 'status', 'design', 'pages', 'tables', 'image tables',
                'text', 'extracted', 'review', 'error')
CONFIRM_FIRST = 'Confirm the design first.'
CODEBOOK_TYPES = ('.xlsx', '.csv', '.yml', '.yaml')
ALL_PAPERS = 'all'
ARM_ROLES = ('treatment', 'control', 'other')
NOT_ARM_REASONS = ('participant characteristic', 'statistic, total, or model row',
                   'set to other at extraction')
STOP_CHECKS = ('arm_role_unsettled', 'test_statistic_unpaired', 'no_comparisons', 'missing_control',
               'several_controls', 'control_at_other_time', 'no_method', 'n_missing', 'n_below_2',
               'sd_not_positive', 'rho_out_of_range', 'r2_out_of_range', 'icc_out_of_range',
               'clusters_missing', 'clusters_invalid', 'clusters_too_few', 'clusters_exceed_n')
HEX_COLOR = re.compile(r'#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})')

try:
    from pubextractor.codebook import safe_color as color_or  # noqa: E402
except ImportError:
    def color_or(value, default):
        return value if isinstance(value, str) and HEX_COLOR.fullmatch(value) else default
ENTRY_COLUMNS = ('arm', 'group', 'outcome', 'time', 'stat', 'value', 'check')
ENTRY_ROWS = 4
VISION_NOTE = ('Reading with a vision model sends the page crop under the license check in the '
               'Model step. Local Ollama needs a vision-capable model: gemma3:4b reads images, '
               'tinyllama does not.')
ROW_COLUMNS = ('field', 'type', 'values', 'search terms', 'stat', 'range', 'definition',
               'question')
RANGE_TEXT = re.compile(r'\s*(-?\d+(?:\.\d+)?)\s*(?:-|to|,)\s*(-?\d+(?:\.\d+)?)\s*')
HELP_PAGES = {'about': 'About', 'codebook': 'Codebook', 'local_model': 'Local model',
              'api_keys': 'API keys', 'license_check': 'License check',
              'copyright': 'Papers and copyright', 'privacy': 'Privacy',
              'recommendations': 'Recommendations',
              'tables_and_effect_sizes': 'Tables and effect sizes',
              'figures': 'Single-case graphs', 'evaluation': 'Agreement',
              'trace': 'Trace and provenance', 'support': 'Support'}
HELP_GROUPS = {'Getting started': ('about', 'codebook', 'recommendations'),
               'Models and permissions': ('local_model', 'api_keys', 'license_check',
                                          'copyright', 'privacy'),
               'Extraction and review': ('tables_and_effect_sizes', 'figures', 'evaluation',
                                         'trace'),
               'Support': ('support',)}
AGREEMENT_KEY = ('paper_id', 'field', 'table', 'row', 'col')
AGREEMENT_VIEW = ('field', 'n', 'agreement %', 'kappa / ICC', 'MAE', 'hallucinations', 'omissions')
GRAPH_MODES = {'calibrate': 'Calibrate', 'points': 'Edit points'}
GRAPH_SHOWS = {'original': 'Original', 'replot': 'Replot'}
FIGURE_KINDS = {'scd_line': 'SCD line', 'bar': 'Bar', 'forest': 'Forest', 'scatter': 'Scatter',
                'flow': 'Flow', 'other': 'Other'}
ERROR_KINDS = {'se': 'SE', 'sd': 'SD', 'ci': '95% CI'}
FIGURE_VIEW = ('paper_id', 'table', 'study', 'arm', 'outcome', 'time', 'stat', 'value', 'status')
SERIES_COLORS = ('#007bc2', '#e5a100', '#2ca02c', '#9467bd', '#6c757d', '#17becf')
GRAPH_NOTE = ('Calibrate: click two x-axis ticks, then two y-axis ticks, and type their values. '
              'Edit points: click to add, click a point to remove.')
TICKS = ('x1', 'x2', 'y1', 'y2')
TICK_AXES = {'x1': 'x (session)', 'y1': 'y (outcome)'}
LOGO = 'data:image/svg+xml,' + quote(files('pubextractor').joinpath('app_pages', 'logo.svg')
                                      .read_text(encoding='utf-8'), safe="=:/' ")
# the logo's mark without its word, square for a browser tab
FAVICON = 'data:image/svg+xml,' + quote(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='2 4 54 52'><rect x='4' y='10' width='18"
    "' height='40' rx='1.5' fill='none' stroke='#264653' stroke-width='2'/><g stroke='#8BA4AD"
    "' stroke-width='1.5'><line x1='8' y1='18' x2='18' y2='18'/><line x1='8' y1='26' x2='18' "
    "y2='26'/><line x1='8' y1='34' x2='18' y2='34'/><line x1='8' y1='42' x2='15' y2='42'/></g"
    "><path d='M25,30 H31 M28.5,26.5 L32,30 L28.5,33.5' fill='none' stroke='#8BA4AD' stroke-w"
    "idth='1.5'/><rect x='35' y='12' width='9' height='9' fill='#E63946'/><rect x='45' y='12'"
    " width='9' height='9' fill='#E63946' fill-opacity='0.4'/><rect x='35' y='25.5' width='9'"
    " height='9' fill='#2A6FA8'/><rect x='45' y='25.5' width='9' height='9' fill='#2A6FA8' fi"
    "ll-opacity='0.4'/><rect x='35' y='39' width='9' height='9' fill='#2A9D8F'/><rect x='45' "
    "y='39' width='9' height='9' fill='#2A9D8F' fill-opacity='0.4'/></svg>"
    , safe="=:/' ")
GRAPH_VIEW = ('study', 'case', 'session', 'phase', 'outcome', 'outcome_type', 'design', 'page',
              'confidence')
GRAPH_CLICK = ("document.getElementById('{plot_id}').on('plotly_click', function (d) {"
               "var p = d.points[0]; Shiny.setInputValue('graph_click', {x: p.x, y: p.y}, "
               "{priority: 'event'}); });")


@cache
def help_html(name):
    """One help page from ``app_pages`` as HTML, with outside links opening in a new tab."""
    name = name if name in HELP_PAGES else 'about'
    text = files('pubextractor').joinpath('app_pages', f"{name}.md").read_text(encoding='utf-8')
    body = MarkdownIt('commonmark', {'html': True}).enable('table').render(text)
    body = body.replace('<table>', '<div class="help-table"><table>').replace(
        '</table>', '</table></div>')
    return re.sub(r'<a href="(https?:|mailto:)', r'<a target="_blank" rel="noopener" href="\1',
                  body.replace('src="logo.svg"', f'src="{LOGO}"'))


def help_view(name, target=None):
    name = name if name in HELP_PAGES else 'about'
    topic = ui.div(
        ui.tags.label('Help topic', for_=f'{target or "help_topic"}_select'),
        ui.tags.select(
            *[ui.tags.optgroup(*[ui.tags.option(HELP_PAGES[key], value=key,
                                               selected=key == name) for key in keys], label=group)
              for group, keys in HELP_GROUPS.items()],
            id=f'{target or "help_topic"}_select', class_='form-select',
            onchange=f"Shiny.setInputValue('{target or 'help_topic'}', this.value, "
                     "{priority: 'event'})"),
        class_='help-topic' + (' compact' if target else ''))
    page = ui.div(ui.HTML(help_html(name)), class_='help-body')
    if target is None:
        return ui.div(topic, page, class_='help-dialog')
    links = [ui.tags.button(title, type='button',
                            class_='help-item' + (' on' if key == name else ''),
                            onclick=f"Shiny.setInputValue('{target}', '{key}', "
                                    "{priority: 'event'})")
             for key, title in HELP_PAGES.items()]
    return ui.div(topic, ui.tags.nav(*links, class_='help-nav', aria_label='Help pages'), page,
                  class_='help-view')


def info_button(page, label=None):
    """Small (i) button that opens one help page."""
    return ui.tags.button(
        ui.span('i', class_='info-i'), label, type='button', class_='info-btn',
        title=f"Help: {HELP_PAGES[page]}", aria_label=f"Help: {HELP_PAGES[page]}",
        onclick=f"event.stopPropagation(); Shiny.setInputValue('help_page', '{page}', "
                "{priority: 'event'})")


class TermEncoder:
    """Fallback sentence encoder: hashed word counts with IDF weights, for retrieval only.

    Used when sentence-transformers is missing. Words are hashed into ``dims``
    buckets with log-scaled counts; IDF weights are learnt from the last call
    with several texts (the chunks) and reused for single queries. It matches
    shared words only, so retrieval is weaker than with a sentence model.
    """

    def __init__(self, dims=2048):
        self.dims = dims
        self.idf = np.ones(dims)

    def _counts(self, texts):
        counts = np.zeros((len(texts), self.dims))
        cells = [(i, zlib.crc32(word.encode('utf-8')) % self.dims)
                 for i, text in enumerate(texts) for word in re.findall(r'\w+', str(text).lower())]
        rows, cols = np.array(cells, dtype=int).reshape(-1, 2).T
        np.add.at(counts, (rows, cols), 1)
        return np.log1p(counts)

    def encode(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        counts = self._counts(texts)
        if len(texts) > 1:
            seen = (counts > 0).sum(axis=0)
            self.idf = np.log((1 + len(texts)) / (1 + seen)) + 1
        weighted = counts * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        return weighted / np.where(norms == 0, 1, norms)


def check_codebook(raw):
    """Codebook from a dict, and the list of problems that stop it loading."""
    try:
        return px.load_codebook(raw), []
    except ValueError as problem:
        return None, re.sub(r'^Codebook problems: ', '', str(problem)).split('; ')


def parse_codebook(text):
    """Codebook from YAML text, and the list of problems that stop it loading."""
    try:
        raw = yaml.safe_load(text or '')
    except Exception as problem:
        return None, [f"YAML could not be read: {problem}"]
    return check_codebook(raw if raw is not None else {})


def field_colors(fields):
    """One palette color per field, by position."""
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(fields or {})}


def format_terms(terms):
    """Search terms as ``value: term, term; value: term``, leaving out values with none."""
    return '; '.join(f"{value}: {', '.join(found)}" for value, found in terms.items() if found)


def parse_terms(text):
    """Search terms from ``value: term, term; value: term``, and the pieces that do not fit."""
    pieces = [piece.partition(':') for piece in (text or '').split(';') if piece.strip()]
    terms = {value.strip(): [term.strip() for term in found.split(',') if term.strip()]
             for value, colon, found in pieces if colon and value.strip() and found.strip()}
    bad = [f"'{(value + colon + found).strip()}' must read value: term, term"
           for value, colon, found in pieces if not colon or not value.strip()]
    return terms, bad


def _number(value):
    return int(float(value)) if float(value).is_integer() else float(value)


def parse_range(text):
    """Range as ``(low, high)`` from text such as ``1-100000``, and whether the text fits."""
    found = RANGE_TEXT.fullmatch(text or '')
    return ((_number(found[1]), _number(found[2])), True) if found else (None, not (text or '').strip())


def _split_values(text):
    return [value.strip() for value in str(text or '').split(',') if value.strip()]


def codebook_rows(codebook):
    """One table row per field, in the columns of the Codebook table."""
    return [{'field': name, 'type': field['type'], 'values': ', '.join(field['values']),
             'search terms': format_terms(field['terms']), 'stat': field.get('stat') or '',
             'range': '-'.join(str(_number(bound)) for bound in field['range'])
             if field.get('range') else '',
             'definition': field.get('definition') or '', 'question': field.get('query') or ''}
            for name, field in codebook.items()]


def _row_field(row, number, keep):
    name = str(row.get('field') or '').strip()
    label = name or f"row {number}"
    terms, bad_terms = parse_terms(row.get('search terms'))
    span, span_ok = parse_range(str(row.get('range') or ''))
    problems = (([] if name else [f"row {number}: field needs a name"])
                + [f"{label}: search terms {piece}" for piece in bad_terms]
                + ([] if span_ok else [f"{label}: range must look like 1-100000"]))
    old = keep.get(name, {})
    field = {'type': str(row.get('type') or '').strip(), 'values': _split_values(row.get('values')),
             'definition': str(row.get('definition') or '').strip(), 'terms': terms,
             'stat': str(row.get('stat') or '').strip() or None, 'range': span,
             'query': str(row.get('question') or '').strip() or None,
             'color': old.get('color'), 'note': old.get('note')}
    return name, field, problems


def rows_to_codebook(rows, keep=None):
    """Codebook from table rows, and the problems that stop it loading.

    ``keep`` is the previous codebook; its color and note per field survive the edit.
    """
    parts = [_row_field(row, number, keep or {}) for number, row in enumerate(rows, 1)]
    names = [name for name, _, _ in parts]
    problems = ([text for _, _, found in parts for text in found]
                + [f"{name}: field name used twice" for name in dict.fromkeys(names)
                   if name and names.count(name) > 1])
    return (None, problems) if problems else check_codebook({name: field for name, field, _ in parts})


def _patterns(book, shown):
    found = [(name, value, px.pattern_from_terms(terms))
             for name, field in (book or {}).items()
             if field['type'] in CATEGORY and (shown is None or name in shown)
             for value, terms in field['terms'].items()]
    return [(name, value, re.compile(pattern, re.IGNORECASE))
            for name, value, pattern in found if pattern]


def term_spans(text, book, shown=None):
    """Codebook terms found in text, earliest and then longest first, never overlapping."""
    found = sorted((match.start(), match.start() - match.end(), name, value)
                   for name, value, pattern in _patterns(book, shown)
                   for match in pattern.finditer(text or '') if match.end() > match.start())
    spans = []
    for start, negative, name, value in found:
        if not spans or start >= spans[-1]['end']:
            spans.append({'start': start, 'end': start - negative, 'field': name, 'value': value})
    return spans


def _entity(text, span, colors):
    color = color_or(colors.get(span['field']), PALETTE[0])
    label = html.escape(f"{span['field']}: {span['value']}")
    return (f'<mark class="entity" style="--entity:{color}">{html.escape(text)}'
            f'<span class="entity-label">{label}</span></mark>')


def marked_text(text, spans, colors):
    """Text as HTML with each span in a colored ``<mark>`` and a FIELD: value chip."""
    starts = [0] + [span['end'] for span in spans]
    pieces = [html.escape(text[begin:span['start']]) + _entity(text[span['start']:span['end']],
                                                                span, colors)
              for begin, span in zip(starts, spans)]
    return ''.join(pieces) + html.escape(text[starts[-1]:])


def term_counts(pages, book, shown=None):
    """Matches per field and value over pages, in codebook order, leaving out zeros."""
    spans = [(span['field'], span['value']) for page in pages
             for span in term_spans(page.get('text') or '', book, shown)]
    return [{'field': name, 'value': value, 'count': spans.count((name, value))}
            for name, field in (book or {}).items() for value in field['values']
            if (name, value) in spans]


def _show(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple)):
        return '; '.join(map(str, value))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _frame(rows, columns):
    return pd.DataFrame([{c: _show(row.get(c)) for c in columns} for row in rows],
                        columns=list(columns))


def is_table_record(record):
    entered = (record.get('layer') in ('human', 'vision', 'cv', 'soma')
               and record.get('table') is not None
               and bool(record.get('stat')))
    return record.get('row') is not None or record.get('field') == 'table' or entered


def open_arms(records):
    """(paper, table, arm, group) of each table arm whose group is not treatment or control."""
    found = {(r['paper_id'], r['table'], r['arm']): r.get('group') for r in records
             if is_table_record(r) and r.get('arm')
             and r.get('group') not in ('treatment', 'control')}
    return [key + (group,) for key, group in found.items()]


def accept_table(records, paper_id, table=None):
    """Records with the review values of one table, or of every table, set to extracted by hand."""
    hit = [is_table_record(r) and (r['paper_id'], r['status']) == (paper_id, 'review')
           and table in (None, r.get('table')) for r in records]
    return ([{**r, 'status': 'extracted', 'layer': 'human'} if h else r
             for r, h in zip(records, hit)], sum(hit))


def review_count(records, paper_id):
    """Values needing review and all values of one paper."""
    statuses = [r['status'] for r in records if r['paper_id'] == paper_id]
    return statuses.count('review'), len(statuses)


def design_label(suggested, confirmed=None):
    return f"{confirmed} ✓" if confirmed else suggested or ''


def queue_row(paper_id, design, confirmed, review, on, evidence=()):
    """One Review queue row: paper, design, Confirm design, review count, and Accept all tables."""
    key = json.dumps(paper_id)
    send = "Shiny.setInputValue('{}', {}, {{priority: 'event'}})".format
    return ui.div(
        ui.span(paper_id, class_='queue-paper', title=paper_id),
        ui.tags.select(*[ui.tags.option(name, value=name, selected=name == design or None)
                         for name in px.DESIGN_TYPES],
                       class_='form-select form-select-sm', aria_label=f"Design of {paper_id}",
                       title='Suggested from: ' + ', '.join(evidence) if evidence else None,
                       onchange=send('queue_design', f"{{paper: {key}, design: this.value}}")),
        ui.tags.button(ui.HTML('&#10003; Confirmed') if confirmed else 'Confirm design',
                       type='button', class_='btn btn-outline-primary btn-sm',
                       onclick=send('queue_confirm', f"{{paper: {key}, design: "
                                    "this.parentNode.querySelector('select').value}")),
        ui.span(f"{review} to review", class_='note'),
        ui.tags.button('Accept all tables', type='button', class_='btn btn-outline-primary btn-sm',
                       onclick=send('queue_accept', key)),
        class_='queue-row' + ' on' * on, data_paper=paper_id, onclick=send('queue_pick', key))


def apply_roles(records, roles, source='human'):
    """Records with the group of each arm in ``roles``, keyed by (paper, table, arm), and its source."""
    return [{**r, 'group': roles[key], 'group_source': source}
            if (key := (r.get('paper_id'), r.get('table'), r.get('arm'))) in roles else r
            for r in records]


def image_badge(grids):
    """Badge text such as '2 image tables', or '' when no table is a picture."""
    count = sum(grid.get('strategy') == 'image' for grid in grids)
    return _plural(count, 'image table') if count else ''


def badge_column(frame):
    """Papers frame with each image-table count shown as a warning badge."""
    return frame.assign(**{'image tables': [ui.span(text, class_='badge text-bg-warning') if text
                                            else '' for text in frame['image tables']]})


def blank_entries(count=ENTRY_ROWS):
    return [dict.fromkeys(ENTRY_COLUMNS, '') for _ in range(count)]


def entry_rows(records):
    """Entry grid rows from a vision model's values, each marked for review."""
    return [{**{key: _show(r.get(key)) for key in ENTRY_COLUMNS[:-1]}, 'check': 'review'}
            for r in records if r.get('stat')]


def replace_entries(records, paper_id, table, new):
    """Records with a coder's earlier entries for one table swapped for ``new``."""
    new = list(new)
    return [r for r in records
            if not (r.get('paper_id') == paper_id and r.get('table') == table
                    and (r.get('layer') in ('human', 'vision') and r.get('row') is None
                         and r.get('stat') or new and r.get('field') == 'table'
                         and r.get('status') == 'not_found'))
            ] + new


def record_key(record):
    return tuple(record.get(k) for k in ('paper_id', 'field', 'table', 'row', 'col'))


def keep_human(new, old):
    """New records with the coder's and vision model's earlier values re-applied."""
    mine = {record_key(r): r for r in old if r.get('layer') in ('human', 'vision')}
    keys = {record_key(r) for r in new}
    entered = {(r.get('paper_id'), r.get('table')) for r in mine.values() if r.get('stat')}
    return ([mine.get(record_key(r), r) for r in new
             if not (r.get('field') == 'table' and r.get('status') == 'not_found'
                     and (r.get('paper_id'), r.get('table')) in entered)]
            + [r for key, r in mine.items() if key not in keys])


def run_tracker(old):
    """A fresh tracker for a new Run, keeping the coder steps ``old`` holds."""
    tracker = px.ProvenanceTracker('fulltext', 'extraction')
    tracker.metadata.update({key: list(value) for key, value in old.metadata.items()
                             if key == 'human'})
    return tracker


def crop_tag(png):
    return ui.tags.img(src='data:image/png;base64,' + base64.b64encode(png).decode('ascii'),
                       alt='Table as printed', class_='table-crop')


def toggle_point(points, x, y, reach=8):
    """Points without those within ``reach`` pixels of the click, or with the click added.

    An added point takes the series of the nearest point, when points carry one.
    """
    near = [p for p in points if np.hypot(p[0] - x, p[1] - y) <= reach]
    closest = min(points, key=lambda p: np.hypot(p[0] - x, p[1] - y), default=(x, y))
    return ([p for p in points if p not in near] if near
            else sorted([*points, (x, y, *closest[2:])], key=lambda p: p[0]))


def place_points(panels, points, design):
    """Panels holding ``points`` (figure pixels), each in the first panel spanning its y.

    Alternating points take the series of the nearest point already in the
    panels; other designs drop the series.
    """
    found = [q for p in panels for q in p['points']]
    marks = [(x, y, *(min(found, key=lambda q: np.hypot(q[0] - x, q[1] - y), default=(x, y))[2:]
                      if design == 'alternating' else ())) for x, y, *_ in points]
    homes = [next((p['panel'] for p in panels if p['top'] <= mark[1] <= p['bottom']),
                  panels[0]['panel']) for mark in marks]
    return [{**p, 'points': [m for m, home in zip(marks, homes) if home == p['panel']]}
            for p in panels]


def series_names(panels):
    """Detected series names across panels, sorted."""
    return sorted({str(q[2]) for p in panels for q in p['points'] if len(q) > 2})


def name_series(panels, names):
    """Panels whose point series are renamed by ``names`` (detected name to new name)."""
    return [{**p, 'points': [(*q[:2], *(names.get(str(s), s) for s in q[2:])) for q in p['points']]}
            for p in panels]


def set_tick(ticks, x, y):
    """Ticks with the click on the first unset of x1, x2, y1, y2; a fifth click starts over."""
    ticks = ticks if len(ticks) < len(TICKS) else {}
    key = next(key for key in TICKS if key not in ticks)
    return {**ticks, key: x if key[0] == 'x' else y}


def _graph_html(figure, height, script=''):
    figure.update_layout(height=height, margin={'t': 24, 'r': 10, 'b': 30, 'l': 40},
                         template='plotly_white')
    return ui.HTML(figure.to_html(full_html=False, include_plotlyjs=False, post_script=script,
                                  config={'displayModeBar': False, 'responsive': True}))


def graph_figure(png, state):
    """The original figure with detected marks and ticks; a click sets ``graph_click``."""
    width, height = px.graphs._open_image(png).size
    points = [(*point, p) for p in state['panels'] for point in p['points']]
    names = list(dict.fromkeys(str(point[2]) for point in points if len(point) > 3))
    figure = go.Figure(go.Image(source='data:image/png;base64,' + base64.b64encode(png).decode(),
                                hovertemplate='px %{x}, %{y}<extra></extra>'))
    figure.add_scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode='markers',
                       marker={'size': 11, 'symbol': 'circle-open', 'line': {'width': 2},
                               'color': [SERIES_COLORS[names.index(str(p[2])) % 6]
                                         if len(p) > 3 else '#007bc2' for p in points]},
                       hovertemplate='point px %{x:.0f}, %{y:.0f}<extra></extra>')
    [figure.add_shape(type='line', x0=x, x1=x, y0=p['top'], y1=p['bottom'],
                      line={'color': '#007bc2', 'dash': 'dash'})
     for p in state['panels'] for x in p['lines']]
    [figure.add_shape(type='line', x0=x0, x1=x1, y0=y, y1=y, line={'color': '#e5a100', 'dash': 'dash'})
     for p in state['panels'] for y, x0, x1 in p['criteria']]
    [figure.add_vline(x=v, line={'color': '#6c757d', 'dash': 'dot'}) if key[0] == 'x'
     else figure.add_hline(y=v, line={'color': '#6c757d', 'dash': 'dot'}) for key, v in state['ticks'].items()]
    figure.update_layout(showlegend=False, dragmode=False, xaxis={'visible': False},
                         yaxis={'visible': False})
    return _graph_html(figure, min(20 + round(600 * height / width), 560), GRAPH_CLICK)


def replot_figure(rows):
    """Saved rows drawn as scdbayes reads them: a panel per case, series colors, phase lines."""
    from plotly.subplots import make_subplots
    cases = list(dict.fromkeys(r['case'] for r in rows))
    names = list(dict.fromkeys(str(r['series']) for r in rows))
    figure = make_subplots(rows=len(cases), cols=1, shared_xaxes=True, subplot_titles=cases,
                           vertical_spacing=0.12 / len(cases))
    runs = [(name, case, list(run)) for case in cases for name in names
            for _, run in groupby([r for r in rows if r['case'] == case
                                   and str(r['series']) == name], key=lambda r: r['phase'])]
    [figure.add_scatter(x=[r['session'] for r in run], y=[r['outcome'] for r in run],
                        mode='lines+markers', name=name, legendgroup=name,
                        showlegend=[n for n, *_ in runs].index(name) == k,
                        marker_color=SERIES_COLORS[names.index(name) % 6],
                        row=cases.index(case) + 1, col=1)
     for k, (name, case, run) in enumerate(runs)]
    [figure.add_vline(x=(a['session'] + b['session']) / 2, row=i + 1, col=1,
                      line={'color': '#007bc2', 'dash': 'dash'})
     for i, case in enumerate(cases) for part in [[r for r in rows if r['case'] == case]]
     for a, b in zip(part, part[1:]) if a['phase'] != b['phase'] and a['series'] == b['series']]
    [figure.add_scatter(x=[r['session'] for r in part], y=[r['criterion'] for r in part],
                        mode='lines', line={'shape': 'hv', 'color': '#e5a100', 'dash': 'dash'},
                        name='criterion', showlegend=i == 0, row=i + 1, col=1)
     for i, case in enumerate(cases)
     for part in [[r for r in rows if r['case'] == case and r.get('criterion', '') != '']]
     if part]
    figure.update_xaxes(title_text='session', row=len(cases), col=1)
    return _graph_html(figure, 120 + 200 * len(cases))


def vision_ready(model, ollama_model, api_model):
    """Whether the Model step names a model that can be sent a table image."""
    named = {'ollama': ollama_model, 'api': api_model}.get(model)
    return bool((named or '').strip())


def parse_edit(column, text, old):
    """Typed value for an edited grid cell; an invalid status keeps the old one."""
    text = str(text).strip()
    if column == 'status':
        return text if text in STATUSES else old
    if column in ('table', 'row', 'col'):
        return int(text) if text.isdigit() else old
    if column != 'value' or not text:
        return text or None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and '.' not in text else number


def ollama_status(base_url):
    """Line under the Ollama box; the server is queried only at a local address."""
    if not px.is_local_endpoint(base_url):
        return 'enter a local Ollama address'
    names = ollama_models(base_url)
    return 'Ollama not reachable' if names is None else 'Installed: ' + (', '.join(names) or 'none')


def model_choices(names=None, current=''):
    """Select choices for the Ollama model: known sizes first, then other installed names."""
    installed = {name.removesuffix(':latest') for name in names or []}
    known = {name: ' · '.join([name, f"{size} GB",
                               'reads images' if name in px.VISION_MODELS else 'text only']
                              + ['installed'] * (name in installed))
             for name, size in px.MODEL_SIZES.items()}
    others = sorted((installed | {current.strip()} - {''}) - set(known))
    return {**known, **{name: ' · '.join([name] + ['installed'] * (name in installed))
                        for name in others}}


def storage_view(check, model):
    """Status and install hint under the Ollama model; red when the disk is too small."""
    low = check['enough'] is False and not check['installed']
    reads = 'reads images' if check['vision'] else 'text only'
    lines = [ui.div(f"{check['message']} · {reads}",
                    class_='note text-danger' if low else 'note')]
    hint = [] if check['installed'] else [ui.div(f"To install: ollama pull {model}",
                                                 class_='note')]
    return ui.TagList(*lines, *hint)


def rows_for(rows, paper_id):
    """Rows of one paper, or all rows for the all-papers choice."""
    return list(rows) if paper_id == ALL_PAPERS else [r for r in rows if r['paper_id'] == paper_id]


def count_papers(found):
    return sum(bool(paper.get('read')) and not paper.get('skipped') for paper in found.values())


def _plural(count, word):
    return f"{count} {word if count == 1 else word + 's'}"


def stop_reason(checks):
    """The check that kept an effect from computing: a stop check if any, else the first."""
    return next((c for c in checks if c in STOP_CHECKS), checks[0] if checks else 'not_computed')


def split_effects(effects, show_all=False):
    """Computed and review effects for the grid (every effect with ``show_all``), and a note
    counting the rest by reason."""
    left = Counter(stop_reason(e['checks']) for e in effects if e['status'] == 'not_computed')
    note = (f"{sum(left.values())} not computed: "
            + ', '.join(f"{count} {reason}" for reason, count in left.most_common())) if left else ''
    return [e for e in effects if show_all or e['status'] != 'not_computed'], note


def human_size(size):
    """File size in B, KB, or MB."""
    size = size or 0
    unit, scale = next((u, s) for u, s in (('MB', 1024 ** 2), ('KB', 1024), ('B', 1))
                       if size >= s or s == 1)
    return f"{size / scale:.3g} {unit}" if scale > 1 else f"{size} B"


def ai_limit(name):
    """A shared-key cap from its environment variable, else the default."""
    variable, default = AI_LIMITS[name]
    try:
        value = int(os.environ.get(variable, ''))
    except ValueError:
        return default
    return value if value > 0 else default


SALT = secrets.token_hex(16)


def client_hash(session):
    """A salted hash of the visitor address; the last proxy hop counts only when trusted."""
    conn = getattr(session, 'http_conn', None)
    hops = [hop.strip() for hop in (conn.headers.get('x-forwarded-for', '') if conn else '').split(',')
            if hop.strip()]
    trusted = os.environ.get('PUBEXTRACTOR_TRUST_PROXY', '').strip().lower() in ('1', 'true', 'yes')
    address = (hops[-1] if trusted and hops
               else conn.client.host if conn is not None and conn.client else 'unknown')
    return hashlib.sha256((SALT + address).encode()).hexdigest()[:16]


def shared_key_problem(client, session_calls, now, calls=AI_CALLS):
    """Reason a shared-key call is refused, or '' after counting it; personal keys are not counted."""
    calls[:] = [call for call in calls if now - call[0] < 86400]
    session_calls[:] = [t for t in session_calls if now - t < 3600]
    hour = [call for call in calls if now - call[0] < 3600]
    problem = ('The shared AI quota for the last 24 hours is used up.' if len(calls) >= ai_limit('day')
               else 'The shared AI quota is used up for this hour.' if len(hour) >= ai_limit('hour')
               else 'This connection reached its shared AI limit for this hour.'
               if sum(call[1] == client for call in hour) >= ai_limit('client')
               else 'This session reached its shared AI limit for this hour.'
               if len(session_calls) >= ai_limit('session') else '')
    if problem:
        return problem + ' Enter a personal key, choose rules only, or try again later.'
    calls.append((now, client))
    session_calls.append(now)
    return ''


def hosted_problem(path, size, uploaded):
    """Reason the demo server refuses an upload, or '' when it fits the demo limits."""
    if uploaded >= HOSTED_FILES:
        return f"the demo takes {HOSTED_FILES} files per session"
    if (size or 0) > HOSTED_BYTES:
        return f"over {HOSTED_BYTES // 1024 ** 2} MB"
    if Path(path).suffix.lower() != '.pdf':
        return ''
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as document:
            pages = len(document.pages)
    except Exception:
        return ''
    return f"{pages} pages; the demo reads up to {HOSTED_PAGES}" if pages > HOSTED_PAGES else ''


def upload_summary(uploads):
    """Short notification text and one item per uploaded file: name, size, type, status, reason."""
    readable = [Path(upload['name']).suffix.lower() in READABLE for upload in uploads]
    items = [{'name': upload['name'], 'size': human_size(upload.get('size')),
              'kind': Path(upload['name']).suffix.lstrip('.').upper(),
              'status': 'ready' if ok else 'skipped', 'reason': '' if ok else SKIP_REASON}
             for upload, ok in zip(uploads, readable)]
    ready = sum(readable)
    skipped = len(items) - ready
    head = f"{_plural(ready, 'file')} ready" if ready else 'No files ready'
    return head + f" · {skipped} skipped" * bool(skipped), items


def codebook_type_problem(name):
    """Message for a codebook file of a type that cannot be imported, else None."""
    suffix = Path(name).suffix
    return (None if suffix.lower() in CODEBOOK_TYPES
            else f"unsupported file type {suffix or '(none)'}: use .xlsx, .csv, or .yml")


def codebook_loaded(name, size, fields):
    return f"Codebook loaded: {name} ({human_size(size)}), {_plural(fields, 'field')}"


def extract_summary(records, papers):
    found = sum(r['status'] in ('extracted', 'review') for r in records)
    review = sum(r['status'] == 'review' for r in records)
    return (f"Extracted {_plural(found, 'value')} from {_plural(papers, 'paper')} "
            f"({review} {'needs' if review == 1 else 'need'} review)")


def effects_summary(effects):
    computed = sum(effect.get('g') is not None for effect in effects)
    return (f"Computed {_plural(computed, 'effect size')} "
            f"({len(effects) - computed} not computed)")


def agreement_table(rows):
    """Per-field agreement for the Export grid, with kappa or ICC in one rounded column."""
    return pd.DataFrame([{
        'field': r['field'], 'n': r['n_pairs'], 'agreement %': round(r['agreement'], 1),
        'kappa / ICC': (f"κ {r['kappa']:.2f}" if r['kappa'] is not None
                        else f"ICC {r['icc']:.2f}" if r['icc'] is not None else ''),
        'MAE': '' if r['mae'] is None else round(r['mae'], 2),
        'hallucinations': r['hallucinations'], 'omissions': r['omissions']} for r in rows],
        columns=list(AGREEMENT_VIEW))


def saved_message(name, rows):
    return f"Saved {name} ({_plural(rows, 'row')})"


def model_message(model, ollama_model, provider, url, names):
    """Notice text and type after a model change; names is None when Ollama is unreachable."""
    if model == 'ollama' and names is None:
        return f"Ollama not reachable at {url}", 'warning'
    return f"Model: {mode_label(model, ollama_model, provider)}", 'message'


def attempt(work, *args, **kwargs):
    """Result of ``work(*args, **kwargs)`` and an empty error, or an empty list and the error."""
    try:
        return work(*args, **kwargs), ''
    except Exception as failure:
        return [], f"{type(failure).__name__}: {failure}"


def notify_loading(text, key):
    ui.notification_show(text, duration=None, id=key, type='message')


def notify(text, kind='message', loading=None):
    """Completion notice for 4 s, or an error that stays until closed; drops the loading notice."""
    if loading:
        ui.notification_remove(loading)
    ui.notification_show(text, type=kind, duration=None if kind == 'error' else 4)


def guarded(label, loading=None):
    """Handler decorator: an unexpected exception becomes a notification, not a dead session."""
    def decorate(handler):
        def fail(failure):
            if isinstance(failure, SilentException):
                raise failure
            notify(f"{label} failed: {type(failure).__name__}: {failure}", 'error', loading)

        @wraps(handler)
        def run(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except Exception as failure:
                fail(failure)

        @wraps(handler)
        async def run_async(*args, **kwargs):
            try:
                return await handler(*args, **kwargs)
            except Exception as failure:
                fail(failure)
        return run_async if inspect.iscoroutinefunction(handler) else run
    return decorate


def ollama_models(base_url, timeout=1.5):
    """Model names an Ollama server lists at /api/tags, or None when it is unreachable."""
    return px.providers.ollama_models(base_url, timeout)


_ENCODERS = {}


def load_encoder():
    """Sentence encoder for retrieval: sentence-transformers when installed, else TermEncoder."""
    if 'model' not in _ENCODERS:
        try:
            from sentence_transformers import SentenceTransformer
            _ENCODERS['model'] = SentenceTransformer(ENCODER_NAME)
        except Exception:
            _ENCODERS['model'] = TermEncoder()
    return _ENCODERS['model']


def example_uploads(folder):
    """The shipped example papers copied into ``folder``, shaped like Shiny file uploads."""
    targets = {name: Path(folder) / name for name in EXAMPLE_PAPERS}
    return [{'name': name, 'datapath': str(path),
             'size': path.write_bytes(EXAMPLES.joinpath(name).read_bytes())}
            for name, path in targets.items()]


def unread_paper(path, size=0):
    """File record for an added paper; nothing is read until extraction starts."""
    path = Path(path)
    skipped = path.suffix.lower() not in READABLE
    return {'paper_id': path.stem, 'name': path.name, 'path': str(path), 'size': size,
            'kind': path.suffix.lower().lstrip('.'), 'pages': [], 'grids': [],
            'error': SKIP_ERROR if skipped else '', 'skipped': skipped, 'has_text': False,
            'read': False}


def paper_status(paper):
    """Status column text: skipped with the reason, ready, read, or error."""
    return ('skipped · ' + SKIP_REASON if paper['skipped'] else 'error' if paper['error']
            else 'read' if paper['read'] else 'ready')


def read_paper(path):
    """Pages, table grids, and a text flag for one PDF, DOCX, or image file."""
    path = Path(path)
    paper = unread_paper(path)
    if paper['skipped']:
        return paper
    try:
        pages, grids = px.read_document(path)
        paper.update(pages=pages if path.suffix.lower() == '.docx'
                     else px.detect_sections(px.detect_page_labels(pages)), grids=grids)
    except Exception as failure:
        paper.update(error=f"{type(failure).__name__}: {failure}")
    paper['has_text'] = any((p.get('text') or '').strip() or p.get('rows')
                            for p in paper['pages'])
    paper['read'] = True
    return paper


def extract_paper(unread, book, paper_id, model, encoder, tracker):
    """One paper read and extracted: the paper, its records, its design suggestion, and errors."""
    paper = unread if unread['read'] else {**read_paper(unread['path']), 'size': unread['size']}
    text, text_error = attempt(px.extract_values, paper['pages'], book, paper_id=paper_id,
                               encoder=encoder, tracker=tracker, **model)
    tables, table_error = attempt(px.extract_table_values, paper['grids'], paper_id=paper_id,
                                  tracker=tracker, **model)
    design, design_error = attempt(px.suggest_design, paper['pages'], paper['grids'],
                                   px.designs.figure_captions(paper['pages']))
    return (paper, text + tables, design or ('group', ['design not read']),
            [f"{paper['name']}: {error}"
             for error in (paper['error'], text_error, table_error, design_error) if error])


def text_sizes(records, pages, paper_id):
    """Arm n records and the total N that one paper's text states, for its arms in ``records``."""
    arms = {r['arm']: r['group'] for r in records if r['paper_id'] == paper_id and r.get('arm')
            and r.get('group') in ('treatment', 'control')}
    found = [r for page in pages
             for r in px.text_arm_n(page.get('text') or '', arms, paper_id,
                                    {'pdf_page': page.get('page'),
                                     'printed_page': page.get('printed_page'),
                                     'section': page.get('section')})]
    return found, px.text_total_n('\n'.join(page.get('text') or '' for page in pages))


def _rule_text(field):
    if field is None:
        return 'field not in the codebook'
    if field['type'] in ('category', 'multi_category'):
        named = sum(bool(terms) for terms in field['terms'].values())
        return (f"terms for {named} of {len(field['values'])} values" if named
                else 'no terms, so no rule')
    if field['type'] in ('integer', 'number'):
        return (f"stat {field['stat']} within {field['range'] or 'any range'}" if field['stat']
                else 'no stat, so no rule')
    return 'text fields have no rule'


def _step(label, value, problem=False):
    return {'label': label, 'value': value, 'problem': problem}


def _text_steps(record, field):
    layer, status, quote = record['layer'], record['status'], record.get('quote') or ''
    regex = {
        'regex': _step('Regex', f"hit: {_show(record['value'])} · "
                       f"{record.get('candidates', 0)} candidate(s)", status == 'review'),
        'none': _step('Regex', quote or 'no hit', True),
    }.get(layer, _step('Regex', 'no hit'))
    failed = quote.startswith('model call failed')
    model = {
        'regex': _step('Model', 'not needed: the rule settled it'),
        'none': _step('Model', 'not asked: rules only'),
    }.get(layer, _step('Model', f"refused: {quote}" if status == 'refused'
                       else quote if failed else 'asked', status == 'refused' or failed))
    grounded = {
        ('regex', 'extracted'): _step('Quote', 'sentence from the document'),
        ('regex', 'review'): _step('Quote', 'sentence from the document'),
        ('prompt', 'extracted'): _step('Quote', 'quote found in the cited passage'),
        ('prompt', 'review'): _step('Quote', 'quote not found in the cited passage', True),
        ('prompt', 'not_found'): _step('Quote', 'the model reports it is not stated'),
    }.get((layer, status), _step('Quote', 'no quote'))
    return [_step('Rules', _rule_text(field)), regex, model, grounded]


def _table_steps(record):
    page = f" · p. {record['pdf_page']}" if record.get('pdf_page') else ''
    return [
        _step('Source', f"Table {record.get('table')} · row {record.get('row')} · "
                        f"col {record.get('col')}{page}"),
        _step('Header', f"stat {record.get('stat') or 'unmapped'} · "
                        f"{record.get('time') or 'untimed'}", not record.get('stat')),
        _step('Arm', f"{record.get('arm') or 'none'} as {record.get('group') or 'unsettled'}",
              bool(record.get('arm')) and not record.get('group')),
        _step('Layer', record.get('layer') or ''),
    ]


def trace_steps(record, field=None, human=()):
    """Steps behind one record, each with label, value, and whether it needs a look.

    Coder steps in ``human`` for the record's paper and field close the chain.
    """
    steps = _table_steps(record) if is_table_record(record) else _text_steps(record, field)
    status = _step('Status', f"{record['status']}: {_show(record['value']) or 'no value'}",
                   record['status'] != 'extracted')
    return steps + [status] + [
        _step('Human', f"{px.trace.DONE.get(h['action'], h['action'])} by "
                       f"{h['coder'] or 'an unnamed coder'}")
        for h in human if (h['paper_id'], h['field']) == (record['paper_id'], record['field'])]


def _location(record):
    parts = [f"p. {record['pdf_page']}" if record.get('pdf_page') else '',
             f"printed p. {record['printed_page']}" if record.get('printed_page') else '',
             f"paragraph {record['paragraph']}" if record.get('paragraph') else '',
             record.get('section') or '']
    return ' · '.join(p for p in parts if p) or 'no location'


def _marked(quote, value):
    text = _show(value)
    if not quote or not text or text not in quote:
        return quote or 'no quote'
    head, tail = quote.split(text, 1)
    return ui.TagList(head, ui.tags.mark(text), tail)


def _write(rows, name, columns=COLUMNS):
    path = Path(tempfile.mkdtemp()) / name
    if rows:
        return px.write_csv(rows, path)
    path.write_text(','.join(columns) + '\n', encoding='utf-8')
    return path


def _plotly_dependency():
    folder = Path(plotly.__file__).parent / 'package_data'
    return HTMLDependency('plotly', plotly.__version__, source={'subdir': str(folder)},
                          script={'src': 'plotly.min.js'})


def _plot(figure):
    figure.update_layout(margin={'t': 30, 'r': 10, 'b': 45, 'l': 60},
                         height=figure.layout.height or 360, template='plotly_white')
    return ui.HTML(figure.to_html(full_html=False, include_plotlyjs=False,
                                  config={'displaylogo': False, 'responsive': True}))


CSS = """
body,table,.shiny-data-grid,.shiny-data-grid :is(table,th,td),.form-control,.form-select,.btn{font-size:14px}
#shiny-notification-panel#shiny-notification-panel{width:auto;max-width:480px;padding:0}
#shiny-notification-panel .shiny-notification.shiny-notification{padding:4px 28px 4px 10px;margin:6px 0 0 auto;width:fit-content;max-width:480px;min-height:0;font-size:14px;line-height:1.3}
.shiny-notification-content,.shiny-notification-content-text,.shiny-notification-content-action{margin:0;padding:0}
.shiny-notification-content-action:empty{display:none}
#shiny-notification-panel .shiny-notification.shiny-notification .shiny-notification-close{top:4px;right:6px;width:18px;height:18px;padding:0;font-size:16px;line-height:1}
.tick{display:grid;grid-template-columns:34px 1fr 1fr;gap:6px;align-items:center;margin-bottom:4px}
.tick>div,.pair>div{margin:0!important;width:auto!important}
.tick input[readonly]{border:0;background:none;padding-left:0;color:#6c757d;box-shadow:none}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.row-tight .shiny-input-container{margin-bottom:0}
.row-tight .input-group .btn,.row-tight .input-group .form-control{padding:.25rem .5rem;font-size:14px;line-height:1.5}
.btn-outline-primary:focus:not(:hover){background:none;color:#007bc2}
.mode{flex-shrink:0}
.mode .btn{border-color:#007bc2;color:#007bc2;white-space:nowrap}
.graph-header{display:flex;align-items:center;flex-wrap:wrap;gap:4px}
.graph-header #graph_caption{flex:1}
#graph_tools{display:flex;align-items:center;flex-wrap:nowrap;flex-basis:100%;gap:4px;overflow-x:auto}
.mode .btn:hover{background:#e7f3fa;color:#007bc2}
.mode .btn.active{background:#007bc2;color:#fff;border-color:#007bc2}
.sdot{width:.55rem;height:.55rem;border-radius:50%;flex-shrink:0;display:inline-block}
.topbar{display:flex;align-items:center;gap:18px;padding:0 16px;height:46px;border-bottom:1px solid var(--bs-border-color)}
.help-page .help-body{max-height:none;overflow:visible}
.flow-step.home .n{display:none}
.modal-content:has(.copyright-check) .modal-header{padding:10px 14px;border:0}
.modal-content:has(.copyright-check) .modal-title{font-size:16px;font-weight:600}
.modal-content:has(.copyright-check) .modal-body{padding:0 14px}
.modal-content:has(.copyright-check) .modal-footer{padding:8px 14px;border:0}
.copyright-check .shiny-input-container{margin-bottom:4px;width:100%}
.copyright-check .form-check{margin-bottom:0}
.copyright-check .note{margin:0}
.info-btn{display:inline-flex;align-items:center;gap:4px;border:0;background:none;padding:0 4px;font-size:12px;font-weight:400;color:#007bc2;vertical-align:middle}
.info-i{width:15px;height:15px;border:1px solid currentColor;border-radius:50%;display:inline-grid;place-items:center;font:italic 600 10px/1 Georgia,serif}
.info-btn:hover{text-decoration:underline}
.help-topic{margin-bottom:20px}
.help-topic label{display:block;font-weight:600;margin-bottom:8px}
.help-topic select{min-height:44px;width:100%}
.help-topic.compact{display:none}
.help-table{max-width:100%;overflow-x:auto;margin:6px 0 12px}
.help-table table{margin:0}
.flow-menu{display:none}
.help-view{display:grid;grid-template-columns:200px minmax(0,1fr);gap:20px}
.help-nav{display:flex;flex-direction:column;gap:2px;border-right:1px solid var(--bs-border-color);padding-right:10px}
.help-item{border:0;border-radius:6px;background:none;text-align:left;padding:6px 10px;font-size:14px;color:#495057}
.help-item:hover{background:#f0f4f8}
.help-item.on{background:#007bc2;color:#fff}
.help-body{font-size:14px;line-height:1.55;max-height:70vh;overflow-y:auto;padding-right:8px}
.help-body h1{font-size:20px;font-weight:600;margin:0 0 10px}
.help-body h2{font-size:16px;font-weight:600;margin:18px 0 6px}
.help-body table{border-collapse:collapse;margin:6px 0 12px;width:auto}
.help-body th,.help-body td{border:1px solid var(--bs-border-color);padding:4px 8px;vertical-align:top}
.help-body th{background:#f8f9fa;font-weight:600}
.help-body pre{background:var(--bs-tertiary-bg);border-radius:6px;padding:8px 10px;font-size:13.5px}
.help-body code{font-size:.95em}
.help-body pre code{color:inherit;font-size:inherit;padding:0}
.shell{min-height:calc(100vh - 46px)}
.flow{display:flex;flex-direction:row;gap:2px;min-width:0;overflow-x:auto}
.flow-step{display:flex;gap:6px;align-items:center;padding:5px 8px;border:0;border-radius:6px;background:none;color:#495057;white-space:nowrap;font-size:14px}
.flow-step:hover{background:#f0f4f8}
.flow-step .n{width:22px;height:22px;flex:0 0 22px;border-radius:50%;border:1px solid #adb5bd;display:grid;place-items:center;font-size:12px}
.flow-step.done .n{background:#e7f3fa;border-color:#007bc2;color:#007bc2}
.flow-step.on{background:#e7f3fa;color:#007bc2;font-weight:600}
.flow-step.on .n{background:#007bc2;color:#fff;border-color:#007bc2}
.flow-step .count{margin-left:2px;font-size:12px;color:var(--bs-secondary-color);font-weight:400}
.main{flex:1;min-width:0;padding:12px 16px}
.main .tab-content>.tab-pane{display:none}
.main .tab-content>.active{display:grid;gap:12px}
.main .card{margin-bottom:0;border-radius:6px;min-width:0;max-width:100%}
.main .card-header{padding:8px 12px;font-weight:600}
.main .card-body{padding:10px 12px;gap:6px;min-width:0}
.main .card-footer{padding:6px 12px}
.main .shiny-input-container{margin-bottom:.5rem}
.main .bslib-grid{gap:12px;margin-bottom:0}
.main .alert{padding:6px 10px;margin-bottom:0;width:fit-content}
.main .shiny-data-grid{width:100%;min-width:0;max-width:100%;max-height:420px;overflow:auto;overscroll-behavior-x:contain}
.shiny-data-grid table{min-width:100%}
.shiny-data-grid :is(th,td){white-space:nowrap}
.shiny-data-grid .shiny-input-container{max-width:100%}
.main .nav-pills .nav-link{padding:4px 10px}
.row-tight{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.demo-banner{background:#fff7e6;color:#7a4b00;font-size:13px;padding:4px 16px;border-bottom:1px solid #f0d9a8}
.demo-banner a{color:#7a4b00;text-decoration:underline}
.demo-banner{position:relative;padding-right:32px}
.demo-close{position:absolute;right:8px;top:50%;transform:translateY(-50%);border:0;background:none;color:inherit;font-size:16px;line-height:1;padding:0 4px;cursor:pointer;opacity:.7}
.demo-close:hover,.demo-close:focus-visible{opacity:1}
.topbar .restart{margin-left:auto;padding:1px 10px;font-size:13px}
.arm-roles .arm-table{font-weight:600;margin:10px 0 4px}
.arm-roles .arm-name{min-width:12rem}
.arm-roles details{margin:4px 0 0 0}
.arm-roles summary{font-size:13px;color:var(--bs-secondary-color);cursor:pointer}
.row-tight .shiny-input-container{margin-bottom:0}
.row-tight .shiny-file-input-progress{display:none}
.note{font-size:13px;color:var(--bs-secondary-color)}
.page-title{font-weight:600}
#long_grid td{white-space:nowrap;max-width:28rem;overflow:hidden;text-overflow:ellipsis}
.shiny-data-grid table>:is(thead,tbody)>tr>:is(th,td){padding:4px 8px}
.model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,220px));gap:4px 16px}
.model-grid .shiny-input-container{width:100%}
.verdict-pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--bs-border-color);border-radius:999px;padding:2px 10px;font-size:12px}
.export-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.export-row .btn{width:100%}
.export-row .note{margin-top:4px}
.trace-steps{list-style:none;margin:0;padding:0}
.trace-step{position:relative;display:flex;gap:.75rem;padding:0 0 .6rem 0;cursor:pointer}
.trace-step:not(:last-child)::before{content:"";position:absolute;left:.75rem;top:1.6rem;bottom:.1rem;width:1px;background:var(--bs-border-color)}
.trace-step .dot{flex:0 0 1.5rem;height:1.5rem;border-radius:50%;border:1px solid var(--bs-border-color);background:#fff;font-size:.72rem;display:flex;align-items:center;justify-content:center;color:var(--bs-secondary-color)}
.trace-step.done .dot{background:#e7f3fb;border-color:#9cc9e6;color:#007bc2}
.trace-step.sel .dot{background:#fff4d6;border-color:#e5a100;color:#8a5a00}
.trace-step .body{flex:1;min-width:0;padding:.1rem .5rem;border-radius:.4rem}
.trace-step.sel .body{background:#fff8e6}
.trace-step:hover .body{background:#f5f8fb}
.trace-step .lbl{font-size:.75rem;color:var(--bs-secondary-color)}
.trace-step .val{font-size:.88rem}
mark{background:#ffe08a;padding:0 2px}
.entity-view{line-height:2.3;white-space:pre-wrap;max-height:420px;overflow-y:auto;padding:8px 12px;background:var(--bs-tertiary-bg);border-radius:6px;font-size:.92rem}
mark.entity{background:color-mix(in srgb,var(--entity) 18%,transparent);border-bottom:2px solid var(--entity);padding:.15em .3em;margin:0 .1em;border-radius:.3em;box-decoration-break:clone;-webkit-box-decoration-break:clone}
.entity-label{font-size:.66em;font-weight:700;text-transform:uppercase;vertical-align:middle;margin-left:.4em;padding:.1em .35em;border-radius:.3em;background:var(--entity);color:#fff;letter-spacing:.02em}
.entity-key{display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem;font-size:.8rem}
.table-crop{max-width:100%;height:auto;border:1px solid var(--bs-border-color);border-radius:4px}
.image-caption{font-weight:600;font-size:13px}
#image_entry td{height:28px}
.queue{display:grid;gap:2px}
.queue-row{display:grid;grid-template-columns:minmax(8rem,1fr) 9.5rem 8.5rem 6.5rem 8.5rem;gap:8px;align-items:center;padding:3px 8px;border-radius:6px;cursor:pointer}
.queue-row:hover{background:#f0f4f8}
.queue-row.on{background:#e7f3fa}
.queue-paper{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.upload-list{list-style:none;margin:0;padding:0;font-size:13px;color:var(--bs-secondary-color)}
.upload-list .upload-skipped{color:#dc3545}
#app-splash{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:#fff;transition:opacity .35s ease}
.app-splash-box{text-align:center;max-width:340px;padding:0 16px}
.app-splash-dot{width:42px;height:42px;margin:0 auto 18px;border:3px solid #d6e9f5;border-top-color:#007bc2;border-radius:50%;animation:app-splash-spin .9s linear infinite}
@keyframes app-splash-spin{to{transform:rotate(360deg)}}
.app-splash-title{font-size:17px;font-weight:600;color:#007bc2;margin:0 0 6px}
.app-splash-note{font-size:14px;color:#475569;margin:0;line-height:1.5}
.busy-overlay{position:fixed;inset:0;z-index:9990;display:flex;align-items:center;justify-content:center;background:transparent;pointer-events:none;visibility:hidden;opacity:0;transition:opacity .15s,visibility 0s linear .15s}
html.shiny-busy .busy-overlay{visibility:visible;opacity:1;transition:opacity .15s .3s,visibility 0s .3s}
body.splash-on .busy-overlay{display:none!important}
.fading-circle{position:relative;width:80px;height:80px}
.fading-circle>div{position:absolute;inset:0;transform:rotate(calc(var(--i) * 30deg))}
.fading-circle>div::before{content:"";display:block;margin:0 auto;width:15%;height:15%;background:#007bc2;border-radius:50%;animation:fading-circle 1.2s infinite ease-in-out both;animation-delay:calc(var(--i) * .1s - 1.2s)}
@keyframes fading-circle{0%,39%,100%{opacity:0}40%{opacity:1}}
@media (max-width:1024px){
.help-view{grid-template-columns:minmax(0,1fr);gap:0}
.help-nav{display:none}
.help-topic.compact{display:block}
.help-body{min-width:0}
.help-body pre{max-width:100%;overflow-x:auto}
.help-body img{max-width:100%;height:auto;display:block}
}
@media (max-width:700px){
.topbar{flex-wrap:wrap;height:auto;padding:6px 12px;gap:6px}
.flow-menu{display:inline-flex;align-items:center;min-height:44px;padding:8px 12px;border:1px solid var(--bs-border-color);border-radius:6px;background:none;color:#495057}
.flow{display:none;order:3;width:100%;flex-wrap:wrap;overflow:visible}
.topbar.menu-open .flow{display:flex}
.flow-step{min-height:44px}
.main,.main .card-body{min-width:0}
.row-tight{flex-wrap:wrap}
.row-tight .shiny-input-container{max-width:100%}
.model-grid{grid-template-columns:minmax(0,1fr)}
.pair{grid-template-columns:minmax(0,1fr)}
.shiny-data-grid{max-width:100%;overflow-x:auto}
.flow-step{padding:5px 6px;font-size:13px}
.flow-step .count{display:none}
.main{padding:12px}
.export-row{grid-template-columns:minmax(0,1fr)}
.queue-row{grid-template-columns:1fr 1fr}
}
"""



STEPS = (('papers', 'Papers'), ('codebook', 'Codebook'), ('model', 'Model'),
         ('review', 'Review'), ('export', 'Export'))
REVIEW_VIEWS = (('values', 'Values'), ('tables', 'Tables'), ('trace', 'Trace'),
                ('figures', 'Figures'))


def mode_label(model, ollama_model='', provider=''):
    """Top-bar text for the model mode."""
    return {'ollama': f"Ollama · {ollama_model}", 'api': f"API · {provider}"}.get(model,
                                                                               'Rules only')


def home_button(current):
    """The Home entry above the numbered steps, holding the help pages."""
    return ui.tags.button(ui.span(class_='n'), 'Home', type='button',
                          class_='flow-step home' + (' on' if current else ''),
                          onclick="Shiny.setInputValue('step', 'home', {priority: 'event'}); "
                                  "document.querySelector('.topbar').classList.remove('menu-open'); "
                                  "document.getElementById('flow_menu').setAttribute('aria-expanded', 'false')")


def step_button(index, key, label, done, current, count=None):
    """One step in the step list: number or tick, label, and an optional muted count."""
    return ui.tags.button(
        ui.span(ui.HTML('&#10003;') if done else str(index + 1), class_='n'), label,
        ui.span(str(count), class_='count') if count else None,
        type='button', class_='flow-step' + (' done' if done else '') + (' on' if current else ''),
        onclick=f"Shiny.setInputValue('step', '{key}', {{priority: 'event'}}); "
                "document.querySelector('.topbar').classList.remove('menu-open'); "
                "document.getElementById('flow_menu').setAttribute('aria-expanded', 'false')")


def _segment(key, label, first, target='review_view'):
    return ui.tags.button(
        label, type='button', value=key, class_='btn btn-outline-primary' + (' active' * first),
        onclick="this.parentNode.querySelectorAll('.btn').forEach(b => "
                "b.classList.toggle('active', b === this)); "
                f"Shiny.setInputValue('{target}', this.value)")


def _pane(key, *content):
    return ui.nav_panel(None, *content, value=key)


model_inputs = ui.TagList(
    ui.div(
        ui.input_select('model', 'Model', {'none': 'None (rules only)', 'api': 'Gemini'} if HOSTED
                        else {'none': 'None (rules only)', 'ollama': 'Ollama (this machine)',
                              'api': 'API provider'}),
        ui.panel_conditional(
            "input.model === 'ollama'",
            ui.input_text('ollama_url', 'Ollama base URL', 'http://localhost:11434/v1')),
        ui.panel_conditional(
            "input.model === 'ollama'",
            ui.input_selectize('ollama_model', 'Ollama model', model_choices(),
                               selected='gemma3:4b', options={'create': True})),
        ui.panel_conditional(
            "input.model === 'api'",
            ui.input_select('provider', 'Provider', ['gemini'] if HOSTED
                            else [p for p in px.PROVIDERS if p != 'ollama'])),
        ui.panel_conditional(
            "input.model === 'api'",
            ui.input_text('api_model', 'API model',
                          API_DEFAULTS['gemini' if HOSTED else 'openai'])),
        ui.panel_conditional(
            "input.model === 'api'",
            ui.input_password('api_key', 'Personal key (optional)',
                              placeholder='uses the shared key when empty')) if HOSTED else None,
        ui.input_text('coder', 'Coder', placeholder='initials, optional'),
        class_='model-grid'),
    ui.panel_conditional("input.model === 'ollama'", ui.output_ui('ollama_list')),
    ui.panel_conditional("input.model === 'ollama'", ui.output_ui('storage_line')),
    ui.output_ui('model_status'),
)

LICENSE_INPUTS = ('access', 'license_type', 'ai_terms', 'processing', 'provider_terms', 'region',
                  'research_org')

license_inputs = ui.div(
    ui.input_select('access', 'Access', {'subscription': 'Subscription',
                                         'open_access': 'Open access',
                                         'personal': 'Personal copy'}),
    ui.input_select('license_type', 'License type', LICENSES),
    ui.input_select('ai_terms', 'AI terms', {'unknown': 'Unknown', 'silent': 'Silent',
                                             'allows': 'Allows', 'forbids': 'Forbids'}),
    ui.input_select('processing', 'Processing', {'local': 'Local', 'private_cloud':
                                                 'Private cloud', 'api': 'API'}),
    ui.input_select('provider_terms', 'Provider terms', {'standard': 'Standard',
                                                         'no_training_no_retention':
                                                         'No training, no retention'}),
    ui.input_select('region', 'Region', {'us': 'US', 'eu': 'EU', 'uk': 'UK', 'other': 'Other'}),
    ui.input_switch('research_org', 'Research organization', True),
    class_='model-grid')

home_page = _pane('home', ui.output_ui('home_panel'))

papers_page = _pane(
    'papers',
    ui.div(
        ui.input_select('paper_source', None, PAPER_SOURCES, width='220px'),
        ui.input_action_button('copyright_check', 'Check copyright (optional)') if HOSTED else None,
        ui.panel_conditional(
            "input.paper_source === 'upload'" + (' && !input.copyright_restricted' if HOSTED else ''),
            ui.div(ui.input_file('files', None, multiple=True, accept=list(READABLE),
                                 width='420px', button_label='Add files'),
                   ui.span('PDF, DOCX, or image · several at once', class_='note'),
                   class_='row-tight')),
        class_='row-tight'),
    ui.p('Upload only papers permitted on a hosted service. Subscription papers stay local.',
         class_='note') if HOSTED else None,
    ui.output_ui('upload_list'),
    ui.card(
        ui.card_header('Papers'),
        ui.output_data_frame('papers_grid'),
        ui.card_footer('The License settings apply to every paper in a run.', class_='note')),
)

codebook_page = _pane(
    'codebook',
    ui.p('What to extract from each paper: one row per field. Edit here or import your Excel '
         'sheet.', info_button('codebook'), class_='mb-2'),
    ui.div(
        ui.input_file('codebook_file', None, accept=['.xlsx', '.csv', '.yml', '.yaml'],
                      width='320px', button_label='Import',
                      placeholder='.xlsx, .csv, or .yml'),
        ui.download_button('dl_codebook', 'Export codebook.xlsx',
                           class_='btn-outline-primary btn-sm'),
        ui.download_link('dl_example', 'Download an example sheet'),
        class_='row-tight'),
    ui.output_ui('codebook_problems'),
    ui.output_data_frame('codebook_grid'),
    ui.div(
        ui.input_action_button('cb_add', '+ Add field', class_='btn-outline-primary btn-sm'),
        ui.input_action_button('cb_remove', 'Remove selected row',
                               class_='btn-outline-primary btn-sm'),
        ui.span('Double-click a cell to edit it; click a row to select it.', class_='note'),
        class_='row-tight mt-2'),
)

model_page = _pane(
    'model',
    ui.card(ui.card_header('Model', info_button('local_model', 'Local model'),
                           info_button('api_keys', 'API keys')), model_inputs,
            ui.div(ui.input_action_button('extract_run', 'Run extraction',
                                          class_='btn-primary btn-sm'),
                   ui.output_ui('rerun_note'), class_='row-tight')),
    ui.card(
        ui.card_header('License', info_button('license_check')),
        license_inputs,
        ui.output_ui('verdict'),
        ui.span('Keys load from .env and are never shown.', class_='note')),
)

review_page = _pane(
    'review',
    ui.div(
        ui.span('Review', class_='page-title'),
        ui.div(*[_segment(key, label, i == 0) for i, (key, label) in enumerate(REVIEW_VIEWS)],
               class_='btn-group btn-group-sm ms-2 mode', role='group', aria_label='Review view'),
        ui.div(
            ui.input_select('view_paper', None, [], width='160px'),
            ui.panel_conditional("(input.review_view || 'values') === 'values'",
                                 ui.input_select('view_page', None, [], width='130px')),
            ui.panel_conditional("input.review_view === 'figures'", ui.output_ui('graph_picker')),
            class_='row-tight ms-auto'),
        class_='row-tight'),
    ui.output_ui('review_note'),
    ui.card(ui.div(ui.span('Designs', class_='note'), info_button('recommendations'),
                   ui.span('suggested from each Method; confirm or change', class_='note'),
                   class_='row-tight'),
            ui.output_ui('review_queue', class_='queue')),
    ui.navset_hidden(
        _pane('values',
              ui.card(ui.output_ui('view_key'), ui.output_ui('view_text')),
              ui.card(ui.card_header('Extracted values',
                                     ui.input_select('values_paper', None, {ALL_PAPERS: 'All papers'},
                                                     width='160px')),
                      ui.output_data_frame('long_grid'))),
        _pane('tables',
              ui.card(
                  ui.card_header('Extracted tables', info_button('tables_and_effect_sizes')),
                  ui.span('Click a cell to correct it; an edited value is recorded with layer '
                          'human. Status takes extracted, review, not_found, or refused.',
                          class_='note mb-1'),
                  ui.output_data_frame('table_grid'),
                  ui.div(ui.output_ui('accept_picker'),
                         ui.input_action_button('accept_run', 'Accept table',
                                                class_='btn-outline-primary btn-sm'),
                         class_='row-tight mt-2')),
              ui.card(
                  ui.card_header('Image tables'),
                  ui.output_ui('image_picker'),
                  ui.panel_conditional(
                      'input.image_pick',
                      ui.output_ui('image_view'),
                      ui.span('Type each value with its arm, outcome, time, and stat; group '
                              'is read from the arm label when left blank. Stat takes '
                              + ', '.join(px.STAT_ROLES) + '.', class_='note'),
                      ui.output_data_frame('image_entry'),
                      ui.div(ui.input_action_button('entry_add', '+ Add row',
                                                    class_='btn-outline-primary btn-sm'),
                             ui.input_action_button('entry_save', 'Save values',
                                                    class_='btn-primary btn-sm'),
                             ui.input_action_button('entry_vision', 'Read with vision model',
                                                    class_='btn-outline-primary btn-sm',
                                                    disabled=True),
                             class_='row-tight mt-2'),
                      ui.span(VISION_NOTE, class_='note'))),
              ui.card(
                  ui.card_header('Effect sizes'),
                  ui.output_ui('design_inputs'),
                  ui.output_ui('arm_roles'),
                  ui.div(ui.input_action_button('effects_run', 'Compute effect sizes',
                                                class_='btn-primary btn-sm')),
                  ui.output_data_frame('effects_grid'),
                  ui.div(ui.output_ui('effects_note'),
                         ui.input_switch('effects_all', 'Show all', False), class_='row-tight')),
              ui.output_ui('correlation_card'),
              ui.output_ui('figures_body')),
        _pane('trace',
              ui.navset_pill(ui.nav_panel('Group design', value='group'),
                             ui.nav_panel('Single-case', value='scd'),
                             ui.nav_panel('Correlation', value='cor'),
                             ui.nav_panel('Qualitative', value='qual'), id='design'),
              ui.output_ui('trace_body')),
        _pane('figures',
              ui.div(ui.output_text('graph_available'), class_='visually-hidden', aria_hidden='true'),
              ui.layout_columns(
                  ui.card(
                      ui.card_header(ui.output_text('graph_caption', inline=True),
                                     info_button('figures'),
                                     ui.output_ui('graph_tools'), class_='graph-header'),
                      ui.output_ui('graph_view'),
                      ui.panel_conditional("output.graph_available === 'true'",
                                           ui.span(GRAPH_NOTE, class_='note'))),
                  ui.panel_conditional("output.graph_available === 'true'", ui.div(
                      ui.panel_conditional("input.graph_type !== 'other' && input.graph_type !== 'flow'", ui.card(
                          ui.card_header('Axes', ui.output_ui('graph_status', inline=True)),
                          ui.p('Click two labeled ticks on each axis, then enter their printed '
                               'values. Calibration converts image positions to data values.',
                               class_='note'),
                          *[ui.TagList(ui.div(TICK_AXES[key], class_='note')
                                       if key in TICK_AXES else None,
                                       ui.div(ui.span(f'tick {key[1]}', class_='note'),
                                              ui.tags.input(id=f"px_{key}", type='text',
                                                            readonly=True, placeholder='click graph',
                                                            class_='form-control form-control-sm'),
                                              ui.input_numeric(f"tick_{key}", None, None),
                                              class_='tick'))
                            for key in TICKS])),
                      ui.card(
                          ui.card_header('Study', info_button('recommendations')),
                          ui.input_select('graph_type', None, FIGURE_KINDS, width='100%'),
                          ui.panel_conditional(
                              "input.graph_type === 'scd_line'",
                              ui.div(ui.input_select('graph_design', None,
                                                     {key: key for key in px.DESIGNS}),
                                     ui.input_text('graph_panels', None, placeholder='panels: auto'),
                                     class_='pair'),
                              ui.input_text('graph_cases', None, width='100%',
                                            placeholder='cases by panel, comma separated'),
                              ui.div(ui.input_select('graph_outcome', None,
                                                     {key: key for key in px.OUTCOME_TYPES}),
                                     ui.input_select('graph_goal', None,
                                                     {key: key for key in px.graphs.GOALS}),
                                     class_='pair'),
                              ui.output_ui('graph_design_note'),
                              ui.output_ui('graph_lines'),
                              ui.output_ui('graph_series')),
                          ui.panel_conditional(
                              "input.graph_type === 'bar'",
                              ui.input_text('bar_names', None, width='100%',
                                            placeholder='bars: arm / outcome / time n = N, comma separated'),
                              ui.input_select('bar_error', None, ERROR_KINDS, width='100%')),
                          ui.panel_conditional(
                              "input.graph_type === 'forest'",
                              ui.input_text('forest_rows', None, width='100%',
                                            placeholder='row labels top to bottom; blank reads text')),
                          ui.panel_conditional(
                              "input.graph_type === 'scatter'",
                              ui.input_text('scatter_axes', None, width='100%',
                                            placeholder='x name, y name')),
                          ui.panel_conditional(
                              "input.graph_type === 'flow'",
                              ui.div('n per box from the text layer; none means hand entry.',
                                     class_='note')),
                          ui.panel_conditional(
                              "input.graph_type === 'other'",
                              ui.div('Kept with its caption; no values read.', class_='note')),
                          ui.input_action_button('graph_save', 'Save graph data',
                                                 class_='btn-primary btn-sm')),
                      class_='d-grid gap-3 align-content-start')),
                  col_widths=(8, 4)),
              ui.panel_conditional("output.graph_available === 'true'", ui.card(
                  ui.card_header('Graph data', ui.output_ui('graph_count', inline=True),
                                 ui.panel_conditional("input.graph_type === 'scd_line'",
                                     ui.download_button('dl_graph', 'Download scdbayes CSV',
                                                        class_='btn-outline-primary btn-sm ms-auto'))),
                  ui.output_data_frame('graph_grid')))),
        id='review_page'),
)


def _export_item(output_id, label, text):
    return ui.div(ui.download_button(output_id, label, class_='btn-outline-primary btn-sm',
                                     onclick="Shiny.setInputValue('exported', true, "
                                             "{priority: 'event'})"),
                  ui.div(text, class_='note'))


export_page = _pane(
    'export',
    ui.span('Export', class_='page-title'),
    ui.div(
        _export_item('dl_long', 'Long table', 'One row per value, with page, section, and quote.'),
        _export_item('dl_wide', 'Wide table', 'One row per paper, one column per field.'),
        _export_item('dl_effects', 'Effect rows', 'g, variance, method, and checks per comparison.'),
        _export_item('dl_metafor', 'metafor', 'yi and vi per effect, for metafor::rma.'),
        _export_item('dl_scdbayes', 'scdbayes', 'One row per graph point, for scdbayes.'),
        _export_item('dl_matrices', 'Correlation matrices',
                     'One CSV per study over the constructs, and n.csv, zipped.'),
        _export_item('dl_trace', 'Trace (JSON)', 'Steps behind each value, per paper.'),
        _export_item('dl_methods', 'Methods text', 'PRISMA 2020 items 9 and 11, to edit.'),
        class_='export-row'),
    ui.card(
        ui.card_header('Agreement', info_button('evaluation')),
        ui.div(ui.download_button('dl_coding', 'Coding sheet', class_='btn-outline-primary btn-sm'),
               ui.input_file('coded_file', None, accept=['.xlsx'], width='320px',
                             button_label='Upload', placeholder='filled coding sheet'),
               ui.download_button('dl_agreement', 'agreement.csv',
                                  class_='btn-outline-primary btn-sm'),
               ui.input_file('decisions_file', None, accept=['.json'], width='320px',
                             button_label='Decisions', placeholder='coder decisions JSON'),
               class_='row-tight'),
        ui.output_data_frame('agreement_grid'),
        ui.card_footer('A second coder fills coded_value; a blank cell means not found.',
                       class_='note'),
        class_='mt-3'),
)

QUEUE_JS = """$(document).on('shiny:connected', function () {
  Shiny.addCustomMessageHandler('copyright_restricted', function (restricted) {
    Shiny.setInputValue('copyright_restricted', restricted);
  });
  Shiny.addCustomMessageHandler('queue_mark', function (paper) {
    document.querySelectorAll('.queue-row').forEach(function (row) {
      row.classList.toggle('on', row.dataset.paper === paper);
    });
  });
});"""

SPLASH_JS = """(function () {
  if (document.body) document.body.classList.add('splash-on');
  var done = false, quiet;
  function hideSplash() {
    if (done) return;
    done = true;
    clearTimeout(quiet);
    var s = document.getElementById('app-splash');
    if (document.body) document.body.classList.remove('splash-on');
    if (!s) return;
    s.id = 'app-splash-going';
    s.style.opacity = '0';
    setTimeout(function () { if (s.parentNode) s.parentNode.removeChild(s); }, 350);
  }
  $(document).on('shiny:busy', function () { clearTimeout(quiet); });
  $(document).on('shiny:idle', function () {
    clearTimeout(quiet);
    quiet = setTimeout(hideSplash, 800);
  });
  setTimeout(hideSplash, 30000);
})();"""

splash = ui.div(
    ui.div(ui.div(class_='app-splash-dot'),
           ui.p('Starting pubextractor', class_='app-splash-title'),
           ui.p('Loading the workspace. The first open takes a few seconds.',
                class_='app-splash-note'),
           class_='app-splash-box'),
    id='app-splash')

busy_spinner = ui.div(ui.div(*[ui.div(style=f"--i:{i}") for i in range(12)],
                             class_='fading-circle'),
                      class_='busy-overlay', role='status', aria_label='Working')

app_ui = ui.page_fluid(
    ui.head_content(ui.tags.link(rel='icon', href=FAVICON)),
    ui.tags.style(CSS),
    splash,
    ui.tags.script(ui.HTML(SPLASH_JS)),
    ui.tags.script(ui.HTML(QUEUE_JS)),
    ui.busy_indicators.use(spinners=False, pulse=False),
    busy_spinner,
    _plotly_dependency(),
    ui.div(DEMO_NOTE, ui.tags.a('Run locally', href='https://github.com/mshin77/pubextractor',
                                target='_blank', rel='noopener'), ' for full use.',
           ui.tags.button(ui.HTML('&times;'), type='button', class_='demo-close',
                          aria_label='Close demo note', onclick='this.parentNode.remove()'),
           class_='demo-banner', role='note') if HOSTED else None,
    ui.div(ui.tags.button(ui.output_ui('menu_label', inline=True), type='button',
                          id='flow_menu', class_='flow-menu', aria_expanded='false',
                          aria_controls='step_list',
                          onclick="const open = this.closest('.topbar').classList.toggle('menu-open'); "
                                  "this.setAttribute('aria-expanded', String(open))"),
           ui.output_ui('step_list', class_='flow'),
           ui.tags.button('Start over', type='button', class_='btn btn-outline-secondary btn-sm restart',
                          onclick="if (confirm('Clear all papers, values, and edits and start again?'))"
                                  " location.reload()"),
           class_='topbar'),
    ui.div(
        ui.div(ui.navset_hidden(home_page, papers_page, codebook_page, model_page, review_page,
                                export_page, id='page'),
               class_='main'),
        class_='shell'),
    title='pubextractor',
    class_='p-0',
)


def server(input, output, session):
    workdir = Path(tempfile.mkdtemp(prefix='pubextractor-'))
    session.on_ended(lambda: shutil.rmtree(workdir, ignore_errors=True))
    session_calls, ai_note = [], {}
    client = client_hash(session)
    px.load_env()
    papers = reactive.value({})
    default = parse_codebook(DEFAULT_CODEBOOK)[0]
    codebook = reactive.value(default)
    problems = reactive.value([])
    draft = {'rows': codebook_rows(default), 'keep': default}
    layout = reactive.value(0)
    records = reactive.value([])
    runs = reactive.value(0)
    edits = reactive.value(0)
    effects = reactive.value([])
    shown = {'rows': []}
    skipped = reactive.value({})
    upload_items = reactive.value([])
    verdicts = {'last': None}
    entries, crops, vision_read = {}, {}, set()
    entry_version = reactive.value(0)
    picked = reactive.value(None)
    stage = reactive.value('home')
    exported = reactive.value(False)
    graphs, figure_items, suggested_graphs = {}, {}, set()
    graph_version = reactive.value(0)
    graph_saved = reactive.value({})
    suggested = reactive.value({})
    confirmed = reactive.value({})
    log = {'tracker': px.ProvenanceTracker('fulltext', 'extraction')}
    queue, kick, run, queue_designs = reactive.value([]), reactive.value(0), {}, {}
    logged = {}

    def _human(paper_id, field, action, before, after):
        with reactive.isolate():
            coder = input.coder().strip() or None
        px.log_human(log['tracker'], paper_id, field, action, before, after, coder)

    def _human_once(paper_id, field, action, before, after):
        """A coder decision logged once, and again only when it changes."""
        if logged.get((paper_id, field)) != (action, after):
            logged[(paper_id, field)] = (action, after)
            _human(paper_id, field, action, before, after)

    @reactive.calc
    def current():
        edits()
        return records()

    def _show_papers(found, selected):
        ui.update_select('view_paper', choices=list(found), selected=selected)
        ui.update_select('values_paper', choices={ALL_PAPERS: 'All papers', **{k: k for k in found}},
                         selected=selected)

    @reactive.effect
    @reactive.event(input.files)
    @guarded('Extraction', 'extract')
    async def _read_files():
        await _add_papers(input.files())

    @reactive.effect
    @reactive.event(input.paper_source)
    @guarded('Extraction', 'extract')
    async def _read_examples():
        if input.paper_source() == 'example':
            await _add_papers(example_uploads(tempfile.mkdtemp()), examples=True)

    @reactive.effect
    @reactive.event(codebook, ignore_init=True)
    @guarded('Extraction', 'extract')
    async def _rerun_on_codebook():
        with reactive.isolate():
            rules_only = input.model() == 'none'
        if papers() and rules_only:
            await _extract()

    @reactive.effect
    @reactive.event(input.extract_run)
    @guarded('Extraction', 'extract')
    async def _run_extraction():
        await _extract()

    def _settings():
        return (codebook(), input.model(), input.ollama_model(), input.api_model(),
                input.provider(), *[input[name]() for name in LICENSE_INPUTS])

    last_settings = reactive.value(None)

    @render.ui
    def rerun_note():
        changed = papers() and last_settings() is not None and _settings() != last_settings()
        return (ui.span('Settings or codebook changed since the last extraction; run it again '
                        'to apply them.', class_='note text-warning') if changed else None)

    def _forget(paper_ids):
        """Drop what an earlier upload under the same name left: design, crops, entries, graphs."""
        confirmed.set({k: v for k, v in confirmed().items() if k not in paper_ids})
        records.set([r for r in records() if r.get('paper_id') not in paper_ids])
        graph_saved.set({k: v for k, v in graph_saved().items()
                         if k.split('|')[0] not in paper_ids})
        [store.pop(key) for store in (entries, crops) for key in list(store) if key[0] in paper_ids]
        [graphs.pop(key) for key in list(graphs) if key.split('|')[0] in paper_ids]
        [store.pop(key, None) for store in (figure_items, queue_designs, design_picks)
         for key in paper_ids]
        [store.pop(key) for store in (picks, construct_picks, logged) for key in list(store)
         if key[0] in paper_ids]
        vision_read.difference_update({key for key in vision_read if key[0] in paper_ids})
        kept_roles.difference_update({key for key in kept_roles if key[0] in paper_ids})

    @reactive.effect
    @reactive.event(input.copyright_check)
    def _ask_copyright():
        if HOSTED:
            ui.modal_show(ui.modal(
                ui.div(ui.input_radio_buttons('copyright_pick', 'Which describes these papers?',
                                              COPYRIGHT_CHOICES, selected='unsure'),
                       ui.p('Subscription papers stay on the computer that holds them.',
                            class_='note'), class_='copyright-check'),
                title='Papers and copyright', easy_close=True, size='s',
                footer=ui.TagList(ui.input_action_button('copyright_cancel', 'Cancel'),
                                  ui.input_action_button('copyright_go', 'Continue',
                                                         class_='btn-primary'))))

    @reactive.effect
    @reactive.event(input.copyright_go)
    async def _copyright_go():
        pick = input.copyright_pick()
        ui.modal_remove()
        await session.send_custom_message('copyright_restricted', pick == 'subscription')
        if pick in ('open_access', 'personal'):
            ui.update_select('access', selected=pick)
            notify('Access set to open access. Select the stated license in Model.' if pick == 'open_access'
                   else 'License set to personal copy.')
            return
        notify(STOP_UPLOAD if pick == 'subscription' else
               'Upload only papers permitted on a hosted service. AI access still requires '
               'confirmed license settings.', 'warning')

    @reactive.effect
    @reactive.event(input.copyright_cancel)
    def _cancel_upload():
        ui.modal_remove()

    async def _add_papers(uploads, examples=False):
        if HOSTED and not examples:
            uploads, refused = _hosted_uploads(uploads)
            [notify(f"{name} not added: {reason}.", 'warning') for name, reason in refused]
            if not uploads:
                return
        found, passed = dict(papers()), dict(skipped())
        for upload in uploads:
            target = workdir / Path(upload['name']).name
            shutil.copy(upload['datapath'], target)
            paper = unread_paper(target, upload.get('size'))
            pool, key = (passed, paper['name']) if paper['skipped'] else (found, paper['paper_id'])
            pool[key] = paper
        _forget({key for key, paper in found.items() if paper is not papers().get(key)})
        papers.set(found)
        skipped.set(passed)
        message, items = upload_summary(uploads)
        upload_items.set(items)
        if passed or not found:
            notify(message, 'warning')
        if found:
            await _extract()

    def _hosted_uploads(uploads):
        """Uploads within the demo limits, and the refused ones with their reasons."""
        before = sum(paper['name'] not in EXAMPLE_PAPERS for paper in papers().values())
        problems = [hosted_problem(upload['datapath'], upload.get('size'), before + i)
                    for i, upload in enumerate(uploads)]
        return ([upload for upload, problem in zip(uploads, problems) if not problem],
                [(upload['name'], problem) for upload, problem in zip(uploads, problems) if problem])

    @render.ui
    def upload_list():
        return ui.tags.ul(*[ui.tags.li(f"{item['name']} · {item['size']} · {item['status']}",
                                       title=item['reason'] or None,
                                       class_=f"upload-{item['status']}")
                            for item in upload_items()], class_='upload-list')

    @reactive.effect
    @reactive.event(input.view_paper)
    def _follow_paper():
        ui.update_select('values_paper', selected=input.view_paper())

    @reactive.effect
    @reactive.event(input.model)
    def _match_processing():
        ui.update_select('processing', selected={'api': 'api'}.get(input.model(), 'local'))

    @reactive.effect
    @reactive.event(input.model, input.ollama_url)
    def _model_choices():
        url = input.ollama_url()
        if input.model() != 'ollama' or not px.is_local_endpoint(url):
            return
        current = input.ollama_model()
        ui.update_selectize('ollama_model', choices=model_choices(ollama_models(url), current),
                            selected=current)

    @reactive.effect
    @reactive.event(input.model, ignore_init=True)
    def _model_notice():
        url = input.ollama_url()
        local = input.model() == 'ollama' and px.is_local_endpoint(url)
        text, kind = model_message(input.model(), input.ollama_model(), input.provider(), url,
                                   ollama_models(url) if local else None)
        notify(text, kind)

    @reactive.effect
    def _verdict_notice():
        found = permission()
        text = (f"License verdict: {found['verdict']} · model use "
                f"{'allowed' if found['allowed'] else 'not allowed'}")
        changed = verdicts['last'] not in (None, text)
        verdicts['last'] = text
        if changed:
            notify(text, 'message' if found['allowed'] else 'warning')

    @reactive.effect
    @reactive.event(input.provider)
    def _api_default():
        ui.update_text('api_model', value=API_DEFAULTS.get(input.provider(), ''))

    @reactive.calc
    def permission():
        return px.check_llm_use(access=input.access(), license_type=input.license_type() or None,
                                ai_terms=input.ai_terms(), processing=input.processing(),
                                region=input.region(), provider_terms=input.provider_terms(),
                                research_org=input.research_org())

    @render.ui
    def verdict():
        found = permission()
        allowed = 'allowed' if found['allowed'] else 'not allowed'
        color = '#198754' if found['allowed'] else '#dc3545'
        return ui.div(
            ui.span(ui.span(class_='sdot', style=f"background:{color}"),
                    f"Verdict: {found['verdict']} · model use {allowed} on this route",
                    class_='verdict-pill'),
            ui.div(found['reason'], class_='note mt-1'))

    @render.ui
    def ollama_list():
        return ui.div(ollama_status(input.ollama_url()), class_='note')

    @render.ui
    def storage_line():
        model = (input.ollama_model() or '').strip()
        if not model:
            return None
        return storage_view(px.storage_check(model, base_url=input.ollama_url()), model)

    @render.ui
    def model_status():
        if input.model() == 'none':
            return ui.div('Rules only: tables and regex.', class_='alert alert-secondary py-2 '
                          'px-2 mb-0')
        if input.model() == 'ollama':
            return ui.div('Local: text stays on this machine.', class_='alert alert-success '
                          'py-2 px-2 mb-0')
        key = px.PROVIDERS[input.provider()][1]
        state = 'found' if key and key in os.environ else 'not set'
        shared = ('Personal key in use.' if HOSTED and input.api_key().strip()
                  else 'Shared key with hourly limits.' if HOSTED else f"{key} {state}.")
        return ui.div(f"API: text leaves the machine under the license check. {shared}",
                      class_='alert alert-warning py-2 px-2 mb-0')

    def _model():
        if input.model() == 'none':
            return {}
        personal = input.api_key().strip() if HOSTED else ''
        key_env = px.PROVIDERS[input.provider()][1]
        responder = (px.make_responder('ollama', input.ollama_model(),
                                       base_url=input.ollama_url())
                     if input.model() == 'ollama'
                     else px.make_responder(input.provider(), input.api_model(),
                                            environ={key_env: personal} if personal else None))
        respond, endpoint = responder
        return {'respond': _capped(respond) if HOSTED and not personal else respond,
                'endpoint': endpoint, 'permission': permission()}

    def _capped(respond):
        """The shared-key responder behind the demo caps; a refused call keeps its reason."""
        def capped(prompt, images=None):
            problem = shared_key_problem(client, session_calls, time.time())
            if problem:
                ai_note['problem'] = problem
                raise PermissionError(problem)
            return respond(prompt, images)
        capped.model, capped.provider = respond.model, respond.provider
        return capped

    async def _extract():
        book, chosen = codebook(), papers()
        if not chosen or book is None:
            notify('Add papers and a working codebook first.', 'warning')
            return
        try:
            model = _model()
        except ValueError as failure:
            notify(str(failure), 'error')
            return
        last_settings.set(_settings())
        notify_loading('Loading the passage encoder…', 'extract') if model else None
        await asyncio.sleep(0)
        encoder = await asyncio.to_thread(load_encoder) if model else None
        ui.notification_remove('extract')
        log['tracker'] = run_tracker(log['tracker'])
        if run.get('bar'):
            run.pop('bar').close()
        run.update(book=book, model=model, encoder=encoder, total=len(chosen), found=[],
                   failed=[], old=records(), bar=ui.Progress(min=0, max=len(chosen)))
        suggested.set({})
        effects.set([])
        _show_papers(chosen, list(chosen)[0])
        queue.set(list(chosen))
        kick.set(kick() + 1)

    @reactive.effect
    @guarded('Extraction', 'extract')
    async def _extract_next():
        kick()
        with reactive.isolate():
            waiting, chosen = queue(), papers()
        if not waiting:
            return
        paper_id, done = waiting[0], run['total'] - len(waiting)
        unread = chosen[paper_id]
        run['bar'].set(done, message=f"Reading {unread['name']} ({done + 1} of {run['total']})")
        result, error = await asyncio.to_thread(
            attempt, extract_paper, unread, run['book'], paper_id, run['model'], run['encoder'],
            log['tracker'])
        paper, found, design, failed = result or (unread, [], ('group', ['paper not read']),
                                                  [f"{unread['name']}: {error}"])
        run['found'] += found
        run['failed'] += failed
        with reactive.isolate():
            papers.set({**papers(), paper_id: paper})
            suggested.set({**suggested(), paper_id: design})
            records.set(keep_human(run['found'], run['old']))
            runs.set(runs() + 1)
            queue.set(waiting[1:])
        if len(waiting) > 1:
            reactive.invalidate_later(0.05)
            return
        run.pop('bar').close()
        notify(extract_summary(run['found'], run['total']), loading='extract')
        notify(ai_note.pop('problem'), 'warning') if ai_note.get('problem') else None
        for problem in run['failed']:
            notify(problem, 'error')

    help_shown = reactive.value('about')

    @reactive.effect
    @reactive.event(input.help_page)
    def _open_help():
        help_shown.set(input.help_page())
        ui.modal_show(ui.modal(ui.output_ui('help_panel'), title='Help', size='xl',
                               easy_close=True, footer=ui.modal_button('Close')))

    @reactive.effect
    @reactive.event(input.help_topic)
    def _pick_help():
        help_shown.set(input.help_topic())

    @render.ui
    def help_panel():
        return help_view(help_shown())

    home_shown = reactive.value('about')

    @reactive.effect
    @reactive.event(input.home_pick)
    def _pick_home():
        home_shown.set(input.home_pick())

    @render.ui
    def home_panel():
        return ui.div(help_view(home_shown(), 'home_pick'), class_='help-page')

    @render.ui
    def menu_label():
        return 'Menu · ' + dict((('home', 'Home'), *STEPS)).get(stage(), 'Home')

    @render.ui
    def step_list():
        found = current()
        chosen = (input.model() == 'none' or
                  bool(input.ollama_model() if input.model() == 'ollama' else input.api_model()))
        done = {'papers': bool(count_papers(papers())), 'review': bool(found), 'export': exported(),
                'codebook': bool(found) and codebook() is not None and not problems(),
                'model': bool(found) and chosen}
        counts = {'papers': count_papers(papers()), 'codebook': len(codebook() or {}),
                  'review': sum(r['status'] == 'review' for r in found)}
        return ui.TagList(home_button(stage() == 'home'),
                          *[step_button(i, key, label, done[key], key == stage(), counts.get(key))
                            for i, (key, label) in enumerate(STEPS)])

    @reactive.effect
    @reactive.event(input.step)
    def _go_step():
        stage.set(input.step())
        ui.update_navset('page', selected=input.step())

    @reactive.effect
    @reactive.event(input.review_view)
    def _go_review_view():
        ui.update_navset('review_page', selected=input.review_view())

    @reactive.effect
    @reactive.event(input.exported)
    def _mark_exported():
        exported.set(True)

    @render.data_frame
    def papers_grid():
        found = current()
        extracted = {r['paper_id'] for r in found}
        counts = {paper_id: '{} / {}'.format(
            sum(r['status'] == 'extracted' for r in found if r['paper_id'] == paper_id),
            sum(r['paper_id'] == paper_id for r in found)) for paper_id in papers()}

        def base(paper):
            return {'file': paper['name'], 'size': human_size(paper['size']),
                    'status': paper_status(paper)}

        def detail(paper_id, paper):
            image = paper['kind'] in IMAGE_KINDS
            return {'pages': (len(paper['pages']) if paper['kind'] == 'pdf' or image
                              else f"{len(paper['pages'])} blocks"),
                    'tables': len(paper['grids']),
                    'image tables': image_badge(paper['grids']) if paper_id in extracted else '',
                    'text': 'image' if image else 'ok' if paper['has_text']
                    else 'no extractable text',
                    'extracted': counts[paper_id], 'error': paper['error'],
                    'design': design_label(suggested().get(paper_id, ('',))[0],
                                           confirmed().get(paper_id)),
                    'review': '{} / {}'.format(*review_count(found, paper_id))}

        read = any(paper['read'] for paper in papers().values())
        rows = [{**base(paper), 'paper': paper_id, **(detail(paper_id, paper) if paper['read']
                                                      else {})}
                for paper_id, paper in papers().items()]
        rows += [{**base(paper), 'paper': ''} for paper in skipped().values()]
        frame = _frame(rows, READ_COLUMNS if read else UPLOAD_COLUMNS)
        return render.DataGrid(badge_column(frame) if read else frame, width='100%')

    @render.ui
    def review_queue():
        found, chosen = current(), confirmed()
        with reactive.isolate():
            shown_paper = input.view_paper()
        rows = [queue_row(paper_id, chosen.get(paper_id, queue_designs.get(paper_id, design)),
                          paper_id in chosen,
                          review_count(found, paper_id)[0], paper_id == shown_paper, evidence)
                for paper_id, (design, evidence) in suggested().items()]
        return ui.TagList(*rows) if rows else ui.span('Add papers to review them.', class_='note')

    @reactive.effect
    @reactive.event(input.queue_design)
    def _queue_design():
        queue_designs[input.queue_design()['paper']] = input.queue_design()['design']

    @reactive.effect
    @reactive.event(input.queue_pick)
    def _queue_pick():
        ui.update_select('view_paper', selected=input.queue_pick())

    @reactive.effect
    @reactive.event(input.view_paper)
    async def _queue_mark():
        await session.send_custom_message('queue_mark', input.view_paper())

    @reactive.effect
    @reactive.event(input.queue_confirm)
    def _queue_confirm():
        paper_id, design = input.queue_confirm()['paper'], input.queue_confirm()['design']
        _human(paper_id, 'design', 'confirm', suggested().get(paper_id, (None,))[0], design)
        confirmed.set({**confirmed(), paper_id: design})
        notify(f"{paper_id}: {design} confirmed")

    @reactive.effect
    @reactive.event(input.queue_accept)
    def _queue_accept():
        _accept(input.queue_accept(), None)

    @render.data_frame
    def long_grid():
        frame = _frame(rows_for(px.to_long(current()), input.values_paper()), LONG_COLUMNS)
        column = LONG_COLUMNS.index('field')
        styles = [{'rows': rows, 'cols': [column],
                   'style': {'border-left': f"4px solid {color}", 'font-weight': '600'}}
                  for name, color in colors().items()
                  for rows in [frame.index[frame['field'] == name].tolist()] if rows]
        return render.DataGrid(frame, width='100%', styles=styles)

    @reactive.calc
    def colors():
        return field_colors(codebook())

    def _apply_rows():
        book, found = rows_to_codebook(draft['rows'], draft['keep'])
        problems.set(found)
        if book is not None:
            draft['keep'] = book
            codebook.set(book)

    def _load_draft(book):
        draft.update(rows=codebook_rows(book), keep=book)
        layout.set(layout() + 1)

    @reactive.effect
    @reactive.event(input.codebook_file)
    @guarded('Codebook import', 'codebook')
    def _codebook_import():
        upload = input.codebook_file()[0]
        problem = codebook_type_problem(upload['name'])
        if problem:
            problems.set([problem])
            notify(f"{upload['name']} was not loaded: {problem}", 'error')
            return
        notify_loading(f"Loading {upload['name']}…", 'codebook')
        path = workdir / Path(upload['name']).name
        shutil.copy(upload['datapath'], path)
        reader = (px.read_codebook_table if path.suffix.lower() in ('.xlsx', '.csv')
                  else px.load_codebook)
        try:
            book = reader(path)
        except Exception as failure:
            found = re.sub(r'^Codebook (table )?problems: ', '', str(failure)).split('; ')
            problems.set(found)
            notify(f"{upload['name']} was not loaded: {'; '.join(found)}", 'error', 'codebook')
            return
        with reactive.isolate():
            _load_draft(book)
            _apply_rows()
        notify(codebook_loaded(upload['name'], upload['size'], len(book)), loading='codebook')

    @render.data_frame
    def codebook_grid():
        layout()
        with reactive.isolate():
            frame = _frame(draft['rows'], ROW_COLUMNS)
        styles = [{'rows': [i], 'cols': [0],
                   'style': {'border-left': f"6px solid {color}", 'font-weight': '600'}}
                  for i, color in enumerate(field_colors(range(len(frame))).values())]
        return render.DataGrid(frame, editable=True, selection_mode='row', summary=False,
                               width='100%', styles=styles)

    @codebook_grid.set_patch_fn
    def _edit_codebook_cell(*, patch):
        draft['rows'][patch['row_index']][ROW_COLUMNS[patch['column_index']]] = str(patch['value'])
        _apply_rows()
        return str(patch['value'])

    @reactive.effect
    @reactive.event(input.cb_add)
    def _codebook_add():
        draft['rows'].append({**dict.fromkeys(ROW_COLUMNS, ''), 'type': 'text'})
        layout.set(layout() + 1)
        _apply_rows()

    @reactive.effect
    @reactive.event(input.cb_remove)
    def _codebook_remove():
        picked = set(codebook_grid.cell_selection()['rows'])
        if not picked:
            notify('Select a row to remove.', 'warning')
            return
        draft['rows'] = [row for i, row in enumerate(draft['rows']) if i not in picked]
        layout.set(layout() + 1)
        _apply_rows()

    @render.ui
    def codebook_problems():
        if not problems():
            return None
        return ui.div(ui.div('Not applied; the last working codebook stays in use:',
                             class_='fw-semibold'),
                      ui.tags.ul(*[ui.tags.li(p) for p in problems()], class_='mb-0'),
                      class_='alert alert-danger py-2')

    @render.download(filename='codebook-example.xlsx')
    def dl_example():
        path = Path(tempfile.mkdtemp()) / 'codebook-example.xlsx'
        return str(px.write_codebook_table(default, path))

    @render.download(filename='codebook.xlsx')
    def dl_codebook():
        book = codebook()
        path = px.write_codebook_table(book, Path(tempfile.mkdtemp()) / 'codebook.xlsx')
        notify(f"Saved codebook.xlsx ({_plural(len(book), 'field')})")
        return str(path)

    @reactive.effect
    @reactive.event(input.view_paper)
    def _view_pages():
        pages = papers().get(input.view_paper(), {}).get('pages', [])
        labels = {str(i): (f"p. {page['page']}" if page.get('page') else
                           f"{page.get('kind') or 'block'} {page.get('paragraph') or i + 1}")
                  for i, page in enumerate(pages)}
        ui.update_select('view_page', choices={'all': 'All pages', **labels},
                         selected='0' if labels else 'all')

    def _picked_pages():
        pages = papers().get(input.view_paper(), {}).get('pages', [])
        index = input.view_page() or 'all'
        return pages if not index.isdigit() else pages[int(index):int(index) + 1]

    @render.ui
    def review_note():
        view = input.review_view() if 'review_view' in input else 'values'
        notes = {
            'values': 'Values lists codebook matches alongside the paper text. Review means '
                      'a candidate needs checking. Not found means no supported match was '
                      'detected; inspect the source before recording it as not reported. '
                      'Refused means the selected license or model route prevented extraction.',
            'tables': 'Tables lists detected numeric cells. Check labels and values, then accept '
                      'the table and compute effect sizes. A paper without a detected table may '
                      'report results in text or images; enter image-table values below.',
            'trace': 'Trace shows the evidence and extraction steps for a selected value. '
                     'Use it to check how the value was obtained and why it needs review.',
            'figures': 'Figures shows detected figures from PDF or DOCX papers. Confirm the '
                       'study design, check the figure type, calibrate numeric axes, and save '
                       'data. SCD points export to scdbayes; other figure types appear in Tables.'}
        return ui.p(notes.get(view, notes['values']), class_='note')

    @render.ui
    def view_key():
        found = colors()
        tally = term_counts(_picked_pages(), codebook())
        counts = {name: sum(c['count'] for c in tally if c['field'] == name) for name in codebook()}
        return ui.div(*[ui.span(ui.span(class_='sdot', style=f"background:{found[name]}"), name,
                                ui.span(str(counts.get(name, 0)), class_='note ms-1'),
                                class_='entity-key')
                        for name, field in codebook().items()
                        if field['type'] in CATEGORY],
                      class_='mb-0')

    @render.ui
    def view_text():
        pages = _picked_pages()
        if not pages:
            return ui.p('Add a paper to see codebook terms marked in its text.',
                        class_='note')
        text = '\n\n'.join(page.get('text') or '' for page in pages)
        spans = term_spans(text, codebook())
        return ui.div(ui.HTML(marked_text(text, spans, colors())), class_='entity-view mt-2')

    @render.ui
    def trace_body():
        later = {'scd': 'Single-case graphs are read in Review → Figures.',
                 'cor': 'Correlation tables export with to_matrices(); see Home.'}
        if input.design() in later:
            return ui.div(later[input.design()],
                          class_='alert alert-secondary')
        return ui.layout_columns(
            ui.card(ui.card_header('Steps'), ui.output_ui('trace_fields'),
                    ui.output_ui('trace_flow')),
            ui.card(ui.card_header(ui.output_text('trace_title', inline=True)),
                    ui.output_ui('trace_evidence')),
            col_widths=(5, 7))

    @render.ui
    def trace_fields():
        book = codebook() or {}
        qualitative = {name for name, field in book.items()
                       if field['type'] in ('category', 'multi_category', 'text')}
        choices = {str(i): (f"{r['field']} · table {r['table']}" if is_table_record(r)
                            else r['field'])
                   for i, r in enumerate(current())
                   if r['paper_id'] == input.view_paper()
                   and (input.design() == 'group' or r['field'] in qualitative)}
        if not choices:
            return ui.p('Add papers to trace values.', class_='note')
        return ui.input_select('trace_record', 'Value', choices)

    def _record():
        index = input.trace_record() if 'trace_record' in input else None
        found = current()
        return found[int(index)] if index and int(index) < len(found) else None

    @reactive.calc
    def steps():
        record = _record()
        return (trace_steps(record, (codebook() or {}).get(record['field']),
                            log['tracker'].to_dict().get('human', [])) if record else [])

    @reactive.effect
    @reactive.event(input.trace_record)
    def _reset_step():
        picked.set(None)

    @reactive.effect
    @reactive.event(input.trace_step)
    def _pick_step():
        picked.set(int(input.trace_step()))

    @render.ui
    def trace_flow():
        found = steps()
        if not found:
            return None
        first = next((i for i, s in enumerate(found) if s['problem']), len(found) - 1)
        selected = first if picked() is None else picked()
        items = [ui.tags.li(
            ui.tags.span(ui.HTML('&#10003;') if i < selected else str(i + 1), class_='dot'),
            ui.div(ui.div(s['label'], ui.span('review', class_='badge text-bg-warning ms-1')
                          if s['problem'] else None, class_='lbl'),
                   ui.div(s['value'], class_='val'), class_='body'),
            class_='trace-step ' + ('sel' if i == selected else 'done' if i < selected else ''),
            onclick=f"Shiny.setInputValue('trace_step', {i}, {{priority: 'event'}})")
            for i, s in enumerate(found)]
        return ui.tags.ol(*items, class_='trace-steps')

    @render.text
    def trace_title():
        record = _record()
        return f"{record['field']} · {record['paper_id']}" if record else 'Evidence'

    @render.ui
    def trace_evidence():
        record = _record()
        if record is None:
            return ui.p('No value selected.', class_='note')
        return ui.TagList(
            ui.div(ui.span(record['status'],
                           class_=f"badge {STATUS_BADGES[record['status']]} me-2"),
                   f"value: {_show(record['value']) or 'none'} · layer {record['layer']}"),
            ui.p(_location(record), class_='note mt-2 mb-1'),
            ui.tags.blockquote(_marked(record.get('quote'), record['value']),
                               class_='border-start ps-2'))

    @render.data_frame
    def table_grid():
        runs()
        with reactive.isolate():
            found = records()
        shown['rows'] = [r for r in found if is_table_record(r)
                         and r['paper_id'] == input.view_paper()]
        return render.DataGrid(_frame(shown['rows'], TABLE_COLUMNS), editable=True,
                               width='100%')

    @table_grid.set_patch_fn
    def _edit_cell(*, patch):
        record = shown['rows'][patch['row_index']]
        column = TABLE_COLUMNS[patch['column_index']]
        before = record.get(column)
        record.update({column: parse_edit(column, patch['value'], before), 'layer': 'human'})
        _human(record['paper_id'], record['field'],
               'correct' if column == 'value' else f"set {column}", before, record[column])
        edits.set(edits() + 1)
        return _show(record[column])

    @render.ui
    def accept_picker():
        tables = sorted({r['table'] for r in current() if isinstance(r.get('table'), int)
                         and r['paper_id'] == input.view_paper() and r.get('table') is not None})
        return ui.input_select('accept_table', None, {str(t): f"Table {t}" for t in tables},
                               width='9rem')

    @reactive.effect
    @reactive.event(input.accept_run)
    def _accept_table():
        table = input.accept_table()
        if table:
            _accept(input.view_paper(), int(table))

    def _accept(paper_id, table):
        kept, count = accept_table(records(), paper_id, table)
        for old in (r for r, new in zip(records(), kept) if new is not r):
            _human(old['paper_id'], old['field'], 'confirm', old['value'], old['value'])
        records.set(kept)
        edits.set(edits() + 1)
        notify(f"{count} values accepted")

    def _image_grids():
        paper = papers().get(input.view_paper(), {})
        return {str(grid['table']): grid for grid in paper.get('grids', [])
                if grid.get('strategy') == 'image'}

    def _picked_image():
        grid = _image_grids().get(input.image_pick() if 'image_pick' in input else None)
        return (input.view_paper(), grid) if grid else (None, None)

    def _crop(paper_id, grid):
        key = (paper_id, grid['table'])
        if key not in crops:
            crops[key] = px.table_image(papers()[paper_id]['path'], grid['pdf_page'], grid['bbox'])
        return crops[key]

    def _entry_rows(paper_id, grid):
        return entries.setdefault((paper_id, grid['table']), blank_entries())

    @render.ui
    def image_picker():
        runs()
        found = _image_grids()
        if not found:
            return ui.p('No table in this paper is printed as an image.',
                        class_='note')
        choices = {key: f"Table {key} · p. {grid['printed_page'] or grid['pdf_page']}"
                   for key, grid in found.items()}
        return ui.div(ui.input_select('image_pick', None, choices, width='220px'),
                      ui.span(f"{_plural(len(found), 'table')} with no text in the body: enter "
                              'the values by hand or read them with a vision model.',
                              class_='note'),
                      class_='row-tight')

    @render.ui
    def image_view():
        paper_id, grid = _picked_image()
        if grid is None:
            return None
        png, problem = attempt(_crop, paper_id, grid)
        return ui.TagList(
            ui.div(grid['caption'] or f"Table {grid['table']}", class_='image-caption'),
            crop_tag(png) if png else ui.div(f"Page crop failed: {problem}",
                                             class_='alert alert-danger'),
            ui.div(grid['note'], class_='note') if grid['note'] else None)

    @render.data_frame
    def image_entry():
        entry_version()
        paper_id, grid = _picked_image()
        rows = _entry_rows(paper_id, grid) if grid else []
        return render.DataGrid(pd.DataFrame(rows, columns=list(ENTRY_COLUMNS)), editable=True,
                               width='100%')

    @image_entry.set_patch_fn
    def _edit_entry(*, patch):
        paper_id, grid = _picked_image()
        text = str(patch['value']).strip()
        _entry_rows(paper_id, grid)[patch['row_index']][ENTRY_COLUMNS[patch['column_index']]] = text
        return text

    @reactive.effect
    def _vision_switch():
        ui.update_action_button('entry_vision', disabled=not vision_ready(
            input.model(), input.ollama_model(), input.api_model()))

    @reactive.effect
    @reactive.event(input.entry_add)
    def _entry_add():
        paper_id, grid = _picked_image()
        if grid:
            _entry_rows(paper_id, grid).append(dict.fromkeys(ENTRY_COLUMNS, ''))
            entry_version.set(entry_version() + 1)

    @reactive.effect
    @reactive.event(input.entry_save)
    @guarded('Saving values', 'entry')
    def _entry_save():
        paper_id, grid = _picked_image()
        if grid is None:
            notify('Pick an image table first.', 'warning')
            return
        try:
            new = px.manual_table_records(_entry_rows(paper_id, grid), paper_id, grid['table'],
                                          pdf_page=grid['pdf_page'],
                                          printed_page=grid['printed_page'],
                                          caption=f"Table {grid['table']}",
                                          layer='vision' if (paper_id, grid['table'])
                                          in vision_read else 'human')
        except ValueError as problem:
            notify(f"Not saved: {problem}", 'error')
            return
        records.set(replace_entries(records(), paper_id, grid['table'], new))
        _human(paper_id, f"table {grid['table']}", 'enter', None, len(new))
        runs.set(runs() + 1)
        notify(f"Saved {_plural(len(new), 'value')} for table {grid['table']}")

    @reactive.effect
    @reactive.event(input.entry_vision)
    @guarded('Vision read', 'vision')
    def _entry_vision():
        paper_id, grid = _picked_image()
        if grid is None:
            notify('Pick an image table first.', 'warning')
            return
        try:
            model = _model()
        except ValueError as failure:
            notify(str(failure), 'error')
            return
        if not model:
            notify('Choose Ollama or an API provider in the Model step first.', 'warning')
            return
        notify_loading(f"Reading table {grid['table']} with "
                       f"{mode_label(input.model(), input.ollama_model(), input.provider())}…",
                       'vision')
        found = px.read_table_image(_crop(paper_id, grid), model['respond'], model['permission'],
                                    model['endpoint'], paper_id=paper_id, table=grid['table'],
                                    pdf_page=grid['pdf_page'], tracker=log['tracker'])
        refused =[r['quote'] for r in found if r['status'] == 'refused']
        if refused:
            notify(f"Nothing sent: {refused[0]}", 'error', 'vision')
            return
        rows = entry_rows(found)
        flagged = [r['quote'] for r in found if not r.get('stat')]
        if not rows:
            notify(f"No values read: {'; '.join(flagged)}", 'error', 'vision')
            return
        entries[(paper_id, grid['table'])] = rows
        vision_read.add((paper_id, grid['table']))
        entry_version.set(entry_version() + 1)
        dropped = f"; {_plural(len(flagged), 'row')} dropped" if flagged else ''
        notify(f"Read {_plural(len(rows), 'value')} from table {grid['table']}{dropped}. "
               'Check each against the image, then Save values.', loading='vision')

    def _group_papers():
        return [name for name in papers() if confirmed().get(name) == 'group']

    def _group_records():
        names = _group_papers()
        return [r for r in current() if r['paper_id'] in names]

    picks, shown_arms, design_picks, shown_designs = {}, [], {}, []
    kept_roles, roles_version = set(), reactive.value(0)

    @reactive.calc
    def arm_suggestions():
        return px.comparisons.suggest_roles(
            _group_records(), {name: papers()[name]['pages'] for name in _group_papers()})

    @reactive.calc
    def assignment_suggestions():
        return {name: px.designs.suggest_assignment(papers()[name]['pages'])
                for name in _group_papers()}

    @render.ui
    def design_inputs():
        names = _group_papers()
        with reactive.isolate():
            design_picks.update({name: input[f"design_{i}"]() for i, name in enumerate(shown_designs)
                                 if f"design_{i}" in input})
        shown_designs[:] = names
        if not names:
            return ui.p(CONFIRM_FIRST if papers() else 'Add papers to set their designs.',
                        class_='note')
        found = assignment_suggestions()
        return ui.TagList(
            ui.div(ui.span('Assignment', class_='note'), info_button('recommendations'),
                   class_='row-tight'),
            ui.layout_columns(*[ui.div(
                ui.input_select(f"design_{i}", name, {'rct': 'RCT', 'qed': 'QED'},
                                selected=design_picks.get(name, found[name][0])),
                ui.span('suggested from ' + ', '.join(found[name][1]) if found[name][1]
                        else 'no assignment terms found; check the Method', class_='note'))
                for i, name in enumerate(names)], col_widths=3))

    @render.ui
    def arm_roles():
        arms = open_arms(_group_records())
        with reactive.isolate():
            picks.update({key: input[f"arm_role_{i}"]() for i, key in enumerate(shown_arms)
                          if f"arm_role_{i}" in input})
        shown_arms[:] = [arm[:3] for arm in arms]
        roles_version()
        found = arm_suggestions()
        rows = [(paper, table, found.get((paper, table, arm), (None, ''))[1], ui.div(
            ui.span(arm, class_='arm-name'),
            ui.input_select(f"arm_role_{i}", None,
                            {'': 'role?', **{role: role for role in ARM_ROLES}}, width='9rem',
                            selected=picks.get((paper, table, arm),
                                               found.get((paper, table, arm), (None,))[0]
                                               or (group if group in ARM_ROLES else ''))),
            ui.span('✓ confirmed' if (paper, table, arm) in kept_roles
                    else found.get((paper, table, arm), (None, ''))[1] or 'no suggestion: pick a role',
                    class_='note'),
            class_='row-tight'))
            for i, (paper, table, arm, group) in enumerate(arms)]
        if not rows:
            return None
        blocks = [ui.div(ui.div(f"{paper} · table {table}", class_='arm-table'),
                         *[row for p, t, reason, row in rows if (p, t) == (paper, table)
                           and not reason.startswith(NOT_ARM_REASONS)],
                         ui.tags.details(ui.tags.summary(f"{_plural(len(hidden), 'row')} suggested as not arms"),
                                         *hidden) if hidden else None)
                  for paper, table in dict.fromkeys((p, t) for p, t, _, _ in rows)
                  for hidden in [[row for p, t, reason, row in rows if (p, t) == (paper, table)
                                  and reason.startswith(NOT_ARM_REASONS)]]]
        suggested = sum(bool(found.get(arm[:3], (None,))[0]) for arm in arms)
        return ui.div(
            ui.div(ui.span('Arm roles', class_='note'), info_button('recommendations'),
                   ui.span(f"{suggested} of {len(arms)} suggested from the data; "
                           'confirm or change each.', class_='note'),
                   ui.input_action_button('roles_confirm', 'Confirm suggestions',
                                          class_='btn-outline-primary btn-sm'),
                   class_='row-tight'),
            *blocks, class_='mb-2 arm-roles')

    def _shown_roles():
        """The role picked for each arm as last shown, keyed by (paper, table, arm)."""
        return {key: input[f"arm_role_{i}"]() for i, key in enumerate(shown_arms)
                if f"arm_role_{i}" in input}

    @reactive.effect
    @reactive.event(input.roles_confirm)
    def _confirm_roles():
        found = arm_suggestions()
        kept = [(key, role) for key, role in _shown_roles().items()
                if role and role == found.get(key, (None,))[0] and key not in kept_roles]
        [_human_once(paper, f"table {table} arm {arm}", 'confirm', role, role)
         for (paper, table, arm), role in kept]
        kept_roles.update(key for key, _ in kept)
        roles_version.set(roles_version() + 1)
        notify(f"Confirmed {_plural(len(kept), 'suggested role')}")

    @reactive.effect
    @reactive.event(input.effects_run)
    @guarded('Effect sizes', 'effects')
    def _effects():
        if not _group_papers():
            notify(CONFIRM_FIRST, 'warning')
            return
        arms, suggestions = open_arms(_group_records()), arm_suggestions()
        groups = {arm[:3]: arm[3] for arm in arms}
        roles = {key: role for key, role in _shown_roles().items()
                 if key in groups and role and role != groups[key]}
        changed = {key for key, role in roles.items() if role != suggestions.get(key, (None,))[0]}
        kept_roles.difference_update(changed)
        suggested = {key: role for key, role in roles.items()
                     if key not in changed and key not in kept_roles}
        [_human_once(paper, f"table {table} arm {arm}", 'role', groups[(paper, table, arm)], role)
         for (paper, table, arm), role in roles.items() if (paper, table, arm) in changed]
        human = {key: role for key, role in roles.items() if key not in suggested}
        found = apply_roles(apply_roles([r for r in _group_records() if is_table_record(r)],
                                        suggested, 'suggested'), human)
        if not found:
            notify('No table records; add papers with tables.', 'warning')
            return
        notify_loading('Computing effect sizes…', 'effects')
        sizes = {name: text_sizes(found, papers()[name]['pages'], name) for name in _group_papers()}
        studies = {name: {'design': input[f"design_{i}"]() or 'rct',
                          **({'n_total': sizes[name][1]} if sizes[name][1] else {})}
                   for i, name in enumerate(_group_papers())}
        stated = [r for rows, _ in sizes.values() for r in rows]
        try:
            computed = px.effect_sizes(px.build_comparisons(found + stated, studies=studies))
        except ValueError as failure:
            notify(str(failure), 'error', 'effects')
            return
        effects.set(computed)
        notify(effects_summary(computed), loading='effects')

    @render.data_frame
    def effects_grid():
        rows = split_effects(effects(), input.effects_all())[0]
        return render.DataGrid(_frame(px.effect_rows(rows), EFFECT_VIEW), width='100%')

    @render.ui
    def effects_note():
        return ui.span(split_effects(effects())[1], class_='note')

    def _correlation_grids():
        grids = papers().get(input.view_paper(), {}).get('grids', [])
        return [grid for grid in grids if px.designs.is_correlation_table(grid)]

    @render.ui
    def correlation_card():
        paper_id = input.view_paper()
        design = confirmed().get(paper_id)
        if 'correlational' not in (design, suggested().get(paper_id, (None,))[0]):
            return None
        with reactive.isolate():
            typed = input.constructs() if 'constructs' in input else ''
        body = (ui.TagList(
            ui.output_data_frame('correlation_grid'),
            ui.div(ui.span('Constructs', class_='note'), info_button('recommendations'),
                   ui.span('suggested from the labels of every correlational paper; edit the '
                           'list, then check each label', class_='note'), class_='row-tight mt-2'),
            ui.input_text('constructs', None, value=typed or ', '.join(suggested_constructs()),
                          width='100%'),
            ui.output_ui('construct_map'))
            if design == 'correlational' else ui.p(CONFIRM_FIRST, class_='note'))
        return ui.card(ui.card_header('Correlations'), body)

    def _correlational():
        return [name for name in papers() if confirmed().get(name) == 'correlational']

    def _matrices(paper_id):
        grids = papers().get(paper_id, {}).get('grids', [])
        return [px.correlation_matrix(grid) for grid in grids
                if px.designs.is_correlation_table(grid)]

    @reactive.calc
    def suggested_constructs():
        return px.suggest_constructs([m['labels'] for name in _correlational()
                                      for m in _matrices(name)])

    @reactive.calc
    def construct_names():
        typed = input.constructs() if 'constructs' in input else ''
        return [name.strip() for name in typed.split(',') if name.strip()] or suggested_constructs()

    construct_picks, shown_labels = {}, []

    def _construct_rows(paper_id, names):
        return [((paper_id, m['table'], label), found)
                for m in _matrices(paper_id)
                for mapping in [px.suggest_mapping(m['labels'], names)]
                for label, found in mapping.items()]

    @render.ui
    def construct_map():
        names = construct_names()
        rows = _construct_rows(input.view_paper(), names)
        with reactive.isolate():
            construct_picks.update({key: input[f"construct_{i}"]() for i, key in enumerate(shown_labels)
                                    if f"construct_{i}" in input})
        shown_labels[:] = [key for key, _ in rows]
        return ui.div(*[ui.div(
            ui.span(f"table {table} · {label}", class_='arm-name'),
            ui.input_select(f"construct_{i}", None, {'': 'none', **{n: n for n in names}},
                            width='14rem',
                            selected=construct_picks.get((paper, table, label), construct or '')),
            ui.span(reason, class_='note'), class_='row-tight')
            for i, ((paper, table, label), (construct, reason)) in enumerate(rows)],
            class_='arm-roles')

    @render.data_frame
    def correlation_grid():
        rows = px.correlation_records([px.correlation_matrix(grid) for grid in _correlation_grids()],
                                      input.view_paper())
        return render.DataGrid(_frame(rows, px.CORRELATION_COLUMNS), width='100%')

    @render.ui
    def figures_body():
        found = current()
        names = list(dict.fromkeys(r['paper_id'] for r in found))
        status = go.Figure([go.Bar(name=s, x=names, marker_color=STATUS_COLORS[s],
                                   y=[sum(r['paper_id'] == n and r['status'] == s for r in found)
                                      for n in names]) for s in STATUSES])
        status.update_layout(barmode='stack', yaxis_title='Values', title='Status by paper')
        rows = [e for e in px.effect_rows(effects()) if e.get('g') is not None]
        forest = go.Figure(go.Scatter(
            x=[e['g'] for e in rows], y=[e['es_id'] for e in rows], mode='markers',
            marker={'color': '#007bc2', 'size': 10},
            error_x={'type': 'data', 'array': [1.96 * (e.get('se_g') or 0) for e in rows]}))
        forest.update_layout(xaxis_title="Hedges' g (95% CI)", title='Effect sizes',
                             yaxis={'automargin': True})
        return ui.TagList(ui.layout_columns(
            ui.card(_plot(status) if found else ui.p('Add papers to chart statuses.')),
            ui.card(_plot(forest) if rows else ui.p('Compute effect sizes to chart them.')),
            col_widths=(6, 6)))

    def _all_graphs():
        found = {paper_id: figure_items[paper_id] if paper_id in figure_items
                 else figure_items.setdefault(
                     paper_id, attempt(px.figure_images, papers()[paper_id]['path']))
                 for paper_id in papers() if paper_id == input.view_paper()}
        return {f"{paper_id}|{i}": (paper_id, item)
                for paper_id, (items, problem) in found.items() for i, item in enumerate(items)}

    def _picked_graph():
        key = input.graph_pick() if 'graph_pick' in input else None
        paper_id, item = _all_graphs().get(key, (None, None))
        if item is None:
            return None, None, None
        design, count, kind = input.graph_design(), input.graph_panels().strip(), input.graph_type()
        state = graphs.get(key, {'ticks': {}})
        fresh = state.get('read') == (design, count, kind)
        read = None if fresh else [] if kind != 'scd_line' else attempt(
            px.read_graph, item['png'], design, int(count) if count.isdigit() else None)[0]
        graphs[key] = (state if fresh else
                       {**state, 'read': (design, count, kind),
                        'panels': place_points(read, state['points'], design)
                        if read and 'points' in state else read})
        return paper_id, item, graphs[key]

    def _show_ticks(state):
        [ui.update_text(f"px_{key}", value=f"px {state['ticks'][key]:.0f}"
                        if key in state['ticks'] else '') for key in TICKS]

    def _graph_rows():
        return [row for rows in graph_saved().values() for row in rows]

    @reactive.calc
    def graph_calibration():
        graph_version()
        state = _picked_graph()[2] or {'ticks': {}}
        ticks = [(state['ticks'].get(key), input[f"tick_{key}"]()) for key in TICKS]
        return None if any(None in pair for pair in ticks) else attempt(px.calibrate, ticks[:2],
                                                                         ticks[2:])

    @render.ui
    def graph_picker():
        found = _all_graphs()
        choices = {key: f"{paper_id} · Figure {item['figure']}"
                   + (f" · p. {item['pdf_page']}" if item['pdf_page'] else '')
                   for key, (paper_id, item) in found.items()}
        return (ui.input_select('graph_pick', None, choices, width='280px') if found
                else ui.span('No figure found.', class_='note'))

    @render.text
    def graph_available():
        paper_id, item, state = _picked_graph()
        return 'true' if item and paper_id in confirmed() else 'false'

    @render.text
    def graph_caption():
        item = _picked_graph()[1]
        return (item['caption'] or f"Figure {item['figure']}") if item else 'Figure'

    @render.ui
    def graph_tools():
        paper_id, item, state = _picked_graph()
        if not item or paper_id not in confirmed():
            return None
        kind = input.graph_type()
        return ui.TagList(
            ui.div(*[_segment(key, label, key == 'original', 'graph_show')
                     for key, label in GRAPH_SHOWS.items()
                     if key == 'original' or kind == 'scd_line'],
                   class_='btn-group btn-group-sm ms-auto mode', role='group',
                   aria_label='Figure view'),
            ui.div(*[_segment(key, label, key == 'calibrate', 'graph_mode')
                     for key, label in GRAPH_MODES.items()
                     if key == 'calibrate' or kind == 'scd_line'],
                   class_='btn-group btn-group-sm mode', role='group',
                   aria_label='Click mode') if kind not in ('other', 'flow') else None)

    @render.ui
    def graph_view():
        graph_version()
        paper_id, item, state = _picked_graph()
        rows = graph_saved().get(input.graph_pick() if item else None)
        replot = input.graph_type() == 'scd_line' and 'graph_show' in input and input.graph_show() == 'replot'
        paper = papers().get(input.view_paper()) or {}
        problem = figure_items.get(input.view_paper(), ([], ''))[1]
        empty = ('Add a PDF or DOCX paper and run extraction to inspect its figures.' if not paper
                 else 'Standalone images appear in Tables → Image tables. For figure '
                      'digitization, add a PDF or DOCX containing the figure.'
                 if Path(paper['path']).suffix.lower() in px.tables.IMAGE_FILES
                 else 'Figure reading could not complete for this file. Check that the PDF or '
                      'DOCX opens correctly, then retry. This is a reading issue, not evidence '
                      'that the paper contains no figures.' if problem
                 else 'No figure was detected in this paper. It may contain no figures, or its '
                      'layout may prevent detection. Check the original PDF or DOCX; this view '
                      'does not establish that results are missing.')
        return (ui.p(empty, class_='note') if not item
                else ui.p('Figure detected. Confirm this paper’s design in Designs above to '
                          'open the figure and data controls.', class_='note') if paper_id not in confirmed()
                else graph_figure(item['png'], state) if not replot
                else replot_figure(rows) if rows
                else ui.p('Save graph data to replot it.', class_='note'))

    @render.ui
    def graph_status():
        found = graph_calibration()
        calibration, problem = found or (None, '')
        logs = [axis for axis in 'xy' if calibration and calibration[axis]['log']]
        return (ui.span('not calibrated', class_='note ms-auto') if not found
                else ui.span(problem, class_='note text-danger ms-auto') if problem
                else ui.span(f"✓ calibrated · residual "
                             f"{100 * max(calibration['x']['residual'], calibration['y']['residual']):.1f}%"
                             + ''.join(f" · log {axis}" for axis in logs),
                             class_='note text-success ms-auto'))

    @render.ui
    def graph_lines():
        graph_version()
        panels = (_picked_graph()[2] or {'panels': []})['panels']
        lines = [len(p['lines']) for p in panels]
        series = series_names(panels)
        criteria = sum(len(p['criteria']) for p in panels)
        phases = [f"{'AB'[k % 2]}{k // 2 + 1}" for k in range(sum(lines[:1]) + 1)]
        return ui.div(f"Series found: {', '.join(series)}" if series
                      else f"{len(panels)} panels · phase lines {', '.join(map(str, lines))}"
                      if len(panels) > 1
                      else f"Phase lines found: {sum(lines)} → {', '.join(phases)}",
                      f" · criterion lines {criteria}" if criteria else '', class_='note')

    @render.ui
    def graph_series():
        state = _picked_graph()[2] or {}
        names = state.get('names', {})
        return ui.TagList(*[ui.input_text(f"series_{i}", name, value=names.get(name, name),
                                          width='100%')
                            for i, name in enumerate(series_names(state.get('panels') or []))])

    @reactive.effect
    @reactive.event(input.graph_pick, confirmed)
    def _graph_pick():
        paper_id, item, state = _picked_graph()
        _show_ticks(state) if state else None
        key = (input.graph_pick(), confirmed().get(paper_id)) if item else None
        if not item or key in suggested_graphs:
            return
        suggested_graphs.add(key)
        ui.update_select('graph_type', selected=attempt(
            px.figure_type, item['png'], item['caption'], confirmed().get(paper_id))[0]
            or 'scd_line')
        found = scd_suggestions()
        [ui.update_select(f"graph_{name}", selected=found[name][0]) for name in found]

    @reactive.calc
    def scd_suggestions():
        pages = (papers().get(input.view_paper()) or {}).get('pages') or []
        item = _all_graphs().get(input.graph_pick() if 'graph_pick' in input else None,
                                 (None, None))[1] or {'caption': '', 'words': []}
        return {'design': px.suggest_scd_design(pages),
                'outcome': px.suggest_outcome_type(item['caption'], item['words']),
                'goal': px.suggest_goal(pages, item['caption'])}

    @render.ui
    def graph_design_note():
        defaults = {'design': 'ab', 'outcome': 'continuous', 'goal': 'increase'}
        return ui.div(*[ui.div(f"{key}: " + ('from ' + ', '.join(evidence) if evidence
                                              else f"no {key} term found; {defaults[key]} by default"),
                               class_='note')
                        for key, (_, evidence) in scd_suggestions().items()])

    @reactive.effect
    @reactive.event(input.graph_click)
    def _graph_click():
        state = _picked_graph()[2]
        click = input.graph_click()
        editing = input.graph_type() == 'scd_line' and 'graph_mode' in input and input.graph_mode() == 'points'
        if not state or editing and not state['panels']:
            return
        points = toggle_point([q for p in state['panels'] for q in p['points']], click['x'],
                              click['y'])
        state.update({'points': points,
                      'panels': place_points(state['panels'], points, input.graph_design())}
                     if editing else {'ticks': set_tick(state['ticks'], click['x'], click['y'])})
        _show_ticks(state)
        graph_version.set(graph_version() + 1)

    @reactive.effect
    @reactive.event(input.graph_save)
    @guarded('Saving graph data')
    def _graph_save():
        paper_id, item, state = _picked_graph()
        if not item:
            notify('No detected figure is selected. Add a PDF or DOCX containing a figure.',
                   'warning')
            return
        calibration, problem = graph_calibration() or (None, 'click two ticks per axis and '
                                                             'type their values first')
        kind = input.graph_type()
        if kind == 'other':
            notify(f"Figure {item['figure']} kept with its caption; no values read.")
            return
        if kind != 'scd_line':
            _figure_save(paper_id, item, calibration, '' if kind == 'flow' else problem, kind)
            return
        if problem or not state['panels']:
            notify(f"Not saved: {problem or 'no axes found in this figure'}", 'warning')
            return
        cases = [case.strip() for case in input.graph_cases().split(',') if case.strip()]
        state['names'] = {name: input[f"series_{i}"]().strip() or name
                          for i, name in enumerate(series_names(state['panels']))
                          if f"series_{i}" in input}
        rows = px.design_rows(name_series(state['panels'], state['names']), calibration,
                              input.graph_design(), cases,
                              outcome_type=input.graph_outcome(), study=paper_id,
                              y_unit='%' if any('%' in e or 'ercent' in e
                                                for e in scd_suggestions()['outcome'][1]) else '',
                              intervention_goal=input.graph_goal(), page=item['pdf_page'] or '',
                              figure=item['figure'])
        graph_saved.set({**graph_saved(), input.graph_pick(): rows})
        _human(paper_id, f"figure {item['figure']}", 'calibrate', None,
               {'design': input.graph_design(), 'calibration': calibration, 'points': len(rows)})
        notify(f"Saved {_plural(len(rows), 'point')} from Figure {item['figure']}")

    def _figure_save(paper_id, item, calibration, problem, kind):
        if problem:
            notify(f"Not saved: {problem}", 'warning')
            return
        where = {'paper_id': paper_id, 'figure': item['figure'], 'pdf_page': item['pdf_page']}
        named = {key: [part.strip() for part in input[key]().split(',') if part.strip()]
                 for key in ('bar_names', 'forest_rows', 'scatter_axes')}
        new = {'bar': lambda: px.read_bars(item['png'], calibration, named['bar_names'],
                                           input.bar_error(), **where),
               'forest': lambda: px.read_forest(item['png'], calibration, named['forest_rows'],
                                                item['words'], **where),
               'scatter': lambda: px.read_scatter(item['png'], calibration,
                                                  named['scatter_axes'] or ('x', 'y'),
                                                  **where)['records'],
               'flow': lambda: px.read_flow(words=item['words'], **where)}[kind]()
        if not new:
            notify(f"Nothing read from Figure {item['figure']}; enter the values by hand.",
                   'warning')
            return
        table = new[0]['table']
        records.set([r for r in records() if (r.get('paper_id'), r.get('table'))
                     != (paper_id, table)] + new)
        _human(paper_id, table, 'calibrate', None,
               {'type': kind, 'calibration': calibration, 'values': len(new)})
        notify(f"Saved {_plural(len(new), 'value')} from Figure {item['figure']}")

    def _figure_records():
        return [r for r in current() if str(r.get('table') or '').startswith('figure')]

    def _scd_view():
        return 'graph_type' not in input or input.graph_type() == 'scd_line'

    @render.ui
    def graph_count():
        count = len(_graph_rows()) if _scd_view() else len(_figure_records())
        return ui.span((f"{_plural(count, 'point')} · scdbayes columns" if _scd_view()
                        else f"{_plural(count, 'value')} · review records") if count
                       else 'Save graph data to list the rows here.',
                       class_='note ms-2')

    @render.data_frame
    def graph_grid():
        rows = _graph_rows() if _scd_view() else _figure_records()
        return render.DataGrid(_frame(rows, GRAPH_VIEW if _scd_view() else FIGURE_VIEW),
                               width='100%') if rows else None

    def _saved(rows, name, columns=COLUMNS):
        path = _write(rows, name, columns)
        notify(saved_message(name, len(rows)))
        return str(path)

    @render.download(filename='pubextractor-long.csv')
    def dl_long():
        return _saved(px.to_long(current()), 'pubextractor-long.csv')

    @render.download(filename='pubextractor-wide.csv')
    def dl_wide():
        return _saved(px.to_wide(current()), 'pubextractor-wide.csv', ('paper_id',))

    @render.download(filename='pubextractor-effects.csv')
    def dl_effects():
        return _saved(px.effect_rows(effects()), 'pubextractor-effects.csv', px.EFFECT_COLUMNS)

    @render.download(filename='pubextractor-metafor.csv')
    def dl_metafor():
        return _saved(px.to_metafor(effects()), 'pubextractor-metafor.csv', px.METAFOR_COLUMNS)

    @render.download(filename='pubextractor-scdbayes.csv')
    def dl_scdbayes():
        return _saved(_graph_rows(), 'pubextractor-scdbayes.csv', px.SCD_COLUMNS)

    @render.download(filename='pubextractor-scdbayes.csv')
    def dl_graph():
        return _saved(_graph_rows(), 'pubextractor-scdbayes.csv', px.SCD_COLUMNS)

    @render.download(filename='pubextractor-matrices.zip')
    def dl_matrices():
        names = construct_names()
        with reactive.isolate():
            construct_picks.update({key: input[f"construct_{i}"]() for i, key in enumerate(shown_labels)
                                    if f"construct_{i}" in input})
        studies = []
        for paper_id in _correlational():
            found = dict(_construct_rows(paper_id, names))
            [_human_once(paper, f"table {table} label {label}", 'construct', found[key][0],
                         picked or None)
             for key, picked in construct_picks.items() if key in found
             for paper, table, label in [key] if (picked or None) != found[key][0]]
            studies += [{'study': f"{paper_id}-table{m['table']}", 'labels': m['labels'],
                         'matrix': m['matrix'], 'n': m['n'],
                         'mapping': {label: construct for label in m['labels']
                                     for construct in [construct_picks.get(
                                         (paper_id, m['table'], label),
                                         found[(paper_id, m['table'], label)][0])] if construct}}
                        for m in _matrices(paper_id)]
        folder = Path(tempfile.mkdtemp()) / 'matrices'
        folder.mkdir()
        problem = ('No correlational paper with a correlation table is confirmed.' if not studies
                   else attempt(px.to_matrices, studies, names, folder)[1])
        if problem:
            (folder / 'problems.txt').write_text(problem + '\n', encoding='utf-8')
            notify(f"Matrices not written: {problem}", 'error')
            return shutil.make_archive(str(folder), 'zip', folder)
        notify(f"Saved pubextractor-matrices.zip ({len(studies)} "
               f"{'matrix' if len(studies) == 1 else 'matrices'})")
        return shutil.make_archive(str(folder), 'zip', folder)

    @render.download(filename='pubextractor-trace.json')
    def dl_trace():
        graphs = px.trace_graph(current(), effects(), _graph_rows(), log['tracker'])
        path = px.write_trace(graphs, Path(tempfile.mkdtemp()) / 'pubextractor-trace.json')
        notify(f"Saved pubextractor-trace.json ({_plural(len(graphs), 'paper')})")
        return str(path)

    @render.download(filename='pubextractor-methods.txt')
    def dl_methods():
        path = Path(tempfile.mkdtemp()) / 'pubextractor-methods.txt'
        path.write_text(''.join(f"PRISMA 2020 {key}\n{text}\n\n" for key, text
                                in px.prisma_items(log['tracker'], current()).items()),
                        encoding='utf-8')
        notify('Saved pubextractor-methods.txt')
        return str(path)

    agreement = reactive.value([])

    @render.download(filename='coding-sheet.xlsx')
    def dl_coding():
        rows = current()
        path = px.write_coding_sheet(rows, Path(tempfile.mkdtemp()) / 'coding-sheet.xlsx')
        notify(saved_message('coding-sheet.xlsx', len(rows)))
        return str(path)

    @reactive.effect
    @reactive.event(input.coded_file)
    @guarded('Reading the coding sheet')
    def _coded():
        upload = input.coded_file()[0]
        rows = px.evaluate(current(), px.read_coding_sheet(upload['datapath']), key=AGREEMENT_KEY)
        agreement.set(rows)
        notify(f"Compared {_plural(len(rows), 'field')} from {upload['name']}")

    @reactive.effect
    @reactive.event(input.decisions_file)
    @guarded('Reading the decisions')
    def _decisions():
        upload = input.decisions_file()[0]
        updated, steps = px.apply_decisions(records(), px.read_decisions(upload['datapath']),
                                            log['tracker'])
        records.set(updated)
        runs.set(runs() + 1)
        notify(f"Applied {_plural(len(steps), 'decision')} from {upload['name']}")

    @render.data_frame
    def agreement_grid():
        rows = agreement()
        return render.DataGrid(agreement_table(rows), width='100%') if rows else None

    @render.download(filename='agreement.csv')
    def dl_agreement():
        return _saved(agreement(), 'agreement.csv', px.AGREEMENT_COLUMNS)


app = App(app_ui, server)


def run_app(port=8000, launch_browser=True):
    """Start the pubextractor app on this machine."""
    from shiny import run_app as run
    run(app, host='127.0.0.1', port=port, launch_browser=launch_browser)


def main(argv=None):
    """Command-line entry point: ``pubextractor [--port 8000] [--no-browser]``."""
    parser = argparse.ArgumentParser(prog='pubextractor', description='Start the pubextractor app.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    run_app(port=args.port, launch_browser=not args.no_browser)
