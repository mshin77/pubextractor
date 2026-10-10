"""Tables printed as pictures: where they sit, their page crop, and values entered or read from it."""

import io
import json
import logging
from pathlib import Path

from .cells import parse_cell
from .layers import _parse
from .licensing import redact_endpoint
from .rag import UNCHECKED, _check_permission
from .table_roles import GROUP_ROLES, GROUP_RULES, STAT_ROLES, TIMES, match_rule
from .tables import (
    IMAGE_FILES,
    _cell_values,
    _namer,
    _record,
    _typed,
    open_image_file,
    read_pdf_tables,
)

logger = logging.getLogger(__name__)

IMAGE_KEYS = ('table', 'pdf_page', 'printed_page', 'caption', 'note', 'bbox', 'reason')
ENTRY_KEYS = ('arm', 'group', 'outcome', 'time', 'stat', 'value')
CODER_QUOTES = {'human': 'entered by coder', 'vision': 'read by a vision model, confirmed by coder'}

VISION_PROMPT = (
    'The image shows one table from a research paper. List every value it reports for a '
    'group or arm.\n\n'
    'For each value give: arm (the group label as printed), outcome (the measure), time '
    '(one of {times}, or null), stat (one of {stats}), value (the number), and cell_text '
    '(the cell copied exactly as printed, digits, signs, and brackets included).\n'
    'Leave out any value whose stat is not in that list.\n\n'
    'Reply with JSON only: {{"rows": [{{"arm": "...", "outcome": "...", "time": null, '
    '"stat": "...", "value": 0, "cell_text": "..."}}]}}')


def find_image_tables(path):
    """Tables in a PDF whose body is a picture with no text, numbered as ``read_tables`` numbers them.

    Each has ``table``, ``pdf_page``, ``printed_page``, the ``caption`` and
    ``note`` lines, ``bbox`` (from the caption to the note or the next text
    row, within the caption's column), and ``reason``.
    """
    return [{key: table[key] for key in IMAGE_KEYS}
            for table in read_pdf_tables(path) if table['strategy'] == 'image']


def table_image(path, pdf_page, bbox, resolution=200):
    """PNG bytes of one page region, rendered at ``resolution`` dots per inch.

    For an image file, ``bbox`` is in pixels and the crop keeps the file's resolution.
    """
    if Path(path).suffix.lower() in IMAGE_FILES:
        picture = open_image_file(path).crop(tuple(int(v) for v in bbox))
    else:
        picture = _page_crop(path, pdf_page, bbox, resolution)
    buffer = io.BytesIO()
    picture.save(buffer, format='PNG')
    return buffer.getvalue()


def _page_crop(path, pdf_page, bbox, resolution):
    try:
        import pdfplumber
    except ImportError:
        raise ImportError('Page crops require pdfplumber; reinstall pubextractor') from None
    with pdfplumber.open(str(path)) as document:
        if not 1 <= pdf_page <= len(document.pages):
            raise ValueError(f"pdf_page {pdf_page} is outside 1-{len(document.pages)}")
        page = document.pages[pdf_page - 1]
        box = (max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], page.width),
               min(bbox[3], page.height))
        return page.crop(box).to_image(resolution=resolution).original


def _text(value):
    return '' if value is None else ' '.join(str(value).split())


def _entry(row):
    entry = {key: _text(row.get(key)) for key in ENTRY_KEYS + ('cell_text',)}
    return {**entry, **{key: entry[key].lower() for key in ('group', 'time', 'stat')}}


def _fits(entry):
    return (entry['stat'] in STAT_ROLES
            and _cell_values(entry['stat'], parse_cell(entry['value']), True)[1] == 'fits')


def _problem(entry):
    return (f"stat {entry['stat']!r} is not one of {list(STAT_ROLES)}"
            if entry['stat'] not in STAT_ROLES
            else f"group {entry['group']!r} is not one of {list(GROUP_ROLES)}"
            if entry['group'] and entry['group'] not in GROUP_ROLES
            else f"time {entry['time']!r} is not one of {list(TIMES)}"
            if entry['time'] and entry['time'] not in TIMES
            else f"value {entry['value']!r} does not read as {entry['stat']}"
            if not _fits(entry) else None)


def _where(table, pdf_page, printed_page):
    return {'table': table, 'pdf_page': pdf_page, 'printed_page': printed_page, 'section': ''}


def _records(entries, paper_id, where, layer, status):
    arms = [{'group': e['group'] or match_rule(GROUP_RULES, e['arm']), 'label': e['arm']}
            for e in entries]
    name = _namer(arms, [e['outcome'] for e in entries])
    return [_record(where, paper_id,
                    {'field': name(e['outcome'] or None, arm, e['time'] or None, stat),
                     'layer': layer, 'quote': e['quote'], 'group': arm['group'],
                     'arm': e['arm'] or None, 'time': e['time'] or None,
                     'outcome': e['outcome'] or None, 'candidates': 0},
                    stat, _typed(stat, value), status)
            for e, arm in zip(entries, arms)
            for stat, value in _cell_values(e['stat'], parse_cell(e['value']), True)[0]]


def manual_table_records(rows, paper_id, table, pdf_page=None, printed_page=None, caption=None,
                         layer='human'):
    """Records for values a coder typed from a table image, in the ``extract_table_values`` format.

    Each row has ``arm``, ``group`` (treatment, control, other; read from the
    arm label by the group rules when blank), ``outcome``, ``time`` (pre,
    post, follow_up, or blank), ``stat`` (one of ``STAT_ROLES``), and
    ``value``; a ``mean_sd`` value such as ``20 (4)`` gives a mean and an SD.
    Blank rows are skipped. Records have status ``extracted`` and layer
    ``human`` with quote ``entered by coder``, or, for rows a vision model
    read and the coder confirmed, layer ``vision`` with quote ``read by a
    vision model, confirmed by coder``. A bad stat, group, time, or value
    raises ``ValueError`` naming each row by its 1-based position.
    """
    if layer not in CODER_QUOTES:
        raise ValueError(f"layer must be one of {list(CODER_QUOTES)}, not {layer!r}")
    entries = [(number, _entry(row)) for number, row in enumerate(rows, start=1)]
    filled = [(number, {**entry, 'quote': CODER_QUOTES[layer]}) for number, entry in entries
              if any(entry[key] for key in ENTRY_KEYS)]
    problems = [f"row {number}: {problem}" for number, entry in filled
                for problem in [_problem(entry)] if problem]
    if problems:
        raise ValueError(f"{caption or f'table {table}'}: " + '; '.join(problems))
    return _records([entry for _, entry in filled], paper_id,
                    _where(table, pdf_page, printed_page), layer, 'extracted')


def _matches(value, cell):
    given = parse_cell(_text(value)) if not isinstance(value, bool) else {'value': None}
    return (given['value'] is not None and given['value'] == cell['value']
            and given.get('second') in (None, cell['second']))


def _read_problem(entry, value):
    return (f"stat {entry['stat']!r} not in the list" if entry['stat'] not in STAT_ROLES
            else f"time {entry['time']!r} not in the list"
            if entry['time'] and entry['time'] not in TIMES
            else 'value does not match the cell text'
            if not (_fits(entry) and _matches(value, parse_cell(entry['value']))) else None)


def _flag(paper_id, where, quote, status='review', entry=None):
    place = {'field': 'table', 'layer': 'vision', 'quote': quote, 'candidates': 0,
             'arm': (entry or {}).get('arm') or None,
             'outcome': (entry or {}).get('outcome') or None}
    return _record(where, paper_id, place, None, None, status)


def read_table_image(png, respond, permission, endpoint, paper_id=None, table=None,
                     pdf_page=None, tracker=None):
    """Values a vision model reads from a table image, every one left for a coder to confirm.

    The license check runs first, as in ``extract_with_rag``: a refusal gives
    one ``refused`` record and nothing is sent. One prompt goes with the
    image, asking for JSON rows with a stat from ``STAT_ROLES`` and each cell
    copied as printed. A row is kept only when its stat is listed and its
    value parses from its ``cell_text``; others come back flagged with no
    value. Kept values have layer ``vision``, status ``review``, and the cell
    text as quote. The tracker records the verdict and the prompt hash and
    model, never the image. A failed call gives ``review`` with
    ``model call failed: <ExceptionType>``.
    """
    where = _where(table, pdf_page, None)
    try:
        checked = _check_permission(permission, endpoint)
    except (PermissionError, ValueError) as refusal:
        return [_flag(paper_id, where, str(refusal), 'refused')]
    prompt = VISION_PROMPT.format(times=json.dumps(list(TIMES)), stats=json.dumps(list(STAT_ROLES)))
    if tracker is not None:
        tracker.log_permission({**({'verdict': UNCHECKED} if checked is None else checked),
                                'endpoint': redact_endpoint(endpoint) if endpoint else None},
                               paper_id=paper_id)
        tracker.log_prompt(prompt, model=getattr(respond, 'model', None))
    try:
        answer = respond(prompt, images=[png])
    except Exception as failure:
        logger.exception('Model call failed for a table image')
        return [_flag(paper_id, where, f'model call failed: {type(failure).__name__}')]
    rows = _parse(answer).get('rows')
    read = [({**_entry({**row, 'value': row.get('cell_text'), 'group': ''}),
              'quote': _text(row.get('cell_text'))}, row.get('value'))
            for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)]
    checked_rows = [(entry, _read_problem(entry, value)) for entry, value in read]
    kept = [entry for entry, problem in checked_rows if problem is None]
    dropped = [_flag(paper_id, where, f"dropped: {problem}: {entry['stat']} {entry['quote']}",
                     entry=entry)
               for entry, problem in checked_rows if problem]
    records = _records(kept, paper_id, where, 'vision', 'review') + dropped
    return records or [_flag(paper_id, where, 'model returned no readable rows')]
