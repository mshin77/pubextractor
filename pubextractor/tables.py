"""Tables read from documents, each with the page and position it came from."""

import re
from collections import Counter
from itertools import accumulate, takewhile
from pathlib import Path
from types import SimpleNamespace

from .cells import header_n, parse_cell
from .fulltext import (
    X_TOLERANCE,
    _heading_name,
    _height,
    detect_page_labels,
    find_gutter,
    page_text,
    report_empty,
    word_rows,
)
from .table_roles import (
    STAT_ROLES,
    STAT_RULES,
    TIME_RULES,
    VETO,
    find_groups,
    label_residue,
    lone_label,
    map_headers,
    match_rule,
    read_layout,
    subheader_rows,
)

CAPTION = re.compile(r'\s*table\s*\d+', re.IGNORECASE)
NOTE = re.compile(r'\s*notes?\b', re.IGNORECASE)


def _joined(lines):
    return ' '.join(' '.join(line.split()) for line in lines if line.strip())


def _around(above, below):
    """Caption from the last 'Table N' line above onward; note from the first 'Note' line below."""
    starts = [i for i, line in enumerate(above) if CAPTION.match(line)]
    notes = [i for i, line in enumerate(below) if NOTE.match(line)]
    ends = [i for i, line in enumerate(below) if CAPTION.match(line) and notes and i > notes[0]]
    return {'caption': _joined(above[starts[-1]:starts[-1] + 3]) if starts else '',
            'note': _joined(below[notes[0]:ends[0] if ends else None]) if notes else ''}


def _between(lines, top, bottom):
    # text lines can overlap a ruling by a fraction of a point, so the line middle decides
    return [line['text'] for line in lines
            if top < (line['top'] + line['bottom']) / 2 < bottom]


STRATEGIES = (('lines', {}),
              ('text-lines', {'vertical_strategy': 'text', 'horizontal_strategy': 'lines'}),
              ('text', {'vertical_strategy': 'text', 'horizontal_strategy': 'text'}))
RULE_MARGIN = 8
MIN_RULE_WIDTH = 20
ALIGN = 2
ANYWHERE = (-float('inf'), -float('inf'), float('inf'), float('inf'))


def _filled(row):
    return sum(bool(cell) for cell in row)


def _ruled_region(page, bbox):
    near = [edge for edge in page.edges
            if edge['orientation'] == 'h' and edge['x1'] - edge['x0'] > MIN_RULE_WIDTH
            and bbox[1] - RULE_MARGIN <= edge['top'] <= bbox[3] + RULE_MARGIN
            and edge['x0'] < bbox[2] and edge['x1'] > bbox[0]]
    return (None if len(near) < 2 else
            (min(e['x0'] for e in near), min(e['top'] for e in near),
             max(e['x1'] for e in near), max(e['top'] for e in near)))


def _split_cells(row):
    gap = 0.5 * max(word['bottom'] - word['top'] for word in row)
    starts = [0] + [i for i in range(1, len(row)) if row[i]['x0'] - row[i - 1]['x1'] > gap]
    return [{'x0': row[a]['x0'], 'x1': row[b - 1]['x1'],
             'text': ' '.join(w['text'] for w in row[a:b])}
            for a, b in zip(starts, starts[1:] + [len(row)])]


def _center(cell):
    return (cell['x0'] + cell['x1']) / 2


def _span_starts(labels, lower):
    """Left edge of each upper label's span: its own left edge when a lower cell starts there
    and is not centered under it, else midway between its center and the previous label's,
    the first one mirrored."""
    centers = [_center(label) for label in labels]
    rights = [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [lower[-1]['x1']]
    flush = [any(abs(cell['x0'] - label['x0']) <= ALIGN < abs(_center(cell) - _center(label))
                 for cell in lower) for label in labels]
    return [label['x0'] if flush[i] else rights[i - 1] if i else 2 * centers[0] - rights[0]
            for i, label in enumerate(labels)]


def _spanned_bands(labels, lower):
    """Band of each upper label: the first lower header cell, past the stub, under its span."""
    starts = _span_starts(labels, lower)
    ends = starts[1:] + [float('inf')]
    return [next((k for k in range(1, len(lower)) if start <= _center(lower[k]) < end), None)
            for start, end in zip(starts, ends)]


def _band_rows(rows):
    widest = max(len(row) for row in rows)
    full = [row for row in rows if len(row) == widest]
    bands = [(min(row[i]['x0'] for row in full), max(row[i]['x1'] for row in full))
             for i in range(widest)]
    top = rows.index(full[0])

    def band(cell):
        overlaps = [i for i, (x0, x1) in enumerate(bands) if cell['x0'] < x1 and cell['x1'] > x0]
        center = _center(cell)
        return overlaps[0] if overlaps else min(
            range(widest), key=lambda i: abs(center - sum(bands[i]) / 2))

    def placed(r, row):
        spanned = _spanned_bands(row, rows[top]) if r < top else [None] * len(row)
        return [band(cell) if k is None or band(cell) == 0 else k
                for cell, k in zip(row, spanned)]
    places = [placed(r, row) for r, row in enumerate(rows)]
    return [[' '.join(c['text'] for c, k in zip(row, place) if k == i) for i in range(widest)]
            for row, place in zip(rows, places)]


def _text_rows(words, region):
    x0, top, x1, bottom = region
    words = [w for w in words
             if x0 - 1 <= w['x0'] and w['x1'] <= x1 + 1
             and top < (w['top'] + w['bottom']) / 2 < bottom]
    return _band_rows([_split_cells(row) for row in word_rows(words)]) if words else []


def _numeric(rows):
    return any(re.search(r'\d', cell) for row in rows[1:] for cell in row)


def _extract(table):
    # pdfplumber scans every page char per row; the chars inside the table give the same cells
    x0, top, x1, bottom = table.bbox
    chars = [c for c in table.page.chars if x0 <= (c['x0'] + c['x1']) / 2 < x1
             and top <= (c['top'] + c['bottom']) / 2 < bottom]
    return type(table)(SimpleNamespace(chars=chars), table.cells).extract(x_tolerance=X_TOLERANCE)


def _text_tables(page, settings, words):
    if _ruled_region(page, ANYWHERE) is None:
        return []
    found = []
    for table in page.find_tables(settings):
        region = _ruled_region(page, table.bbox)
        raw = [row for row in (_extract(table) if region else [])
               if _filled([c or '' for c in row])]
        tidy = not any('\n' in (cell or '') for row in raw for cell in row)
        rows = ([[(c or '').strip() for c in row] for row in raw] if settings.get(
            'horizontal_strategy') == 'lines' else _text_rows(words, region) if region else [])
        if region and tidy and len(rows) > 1 and len(rows[0]) > 1 and _numeric(rows):
            found.append({'rows': rows, 'bbox': (region[0], region[1], region[2], region[3]),
                          'columns_agree': _filled(rows[0]) == max(map(_filled, rows[1:]))})
    return found


def _page_tables(page, words):
    for strategy, settings in STRATEGIES:
        tables = ([{'rows': rows, 'bbox': table.bbox, 'columns_agree': None}
                   for table in page.find_tables()
                   for rows in [_extract(table)] if rows]
                  if strategy == 'lines' else _text_tables(page, settings, words))
        if tables:
            return [{**table, 'strategy': strategy} for table in tables]
    return []


CAPTION_WORD = re.compile(r'(?i)table$')
CAPTION_NUMBER = re.compile(r'[\dIVX]+[.:]?$')
CAPTION_JOINED = re.compile(r'(?i)table[\dIVX]+[.:]?$')
PROSE_WORDS = 6
LABEL_WORDS = 4
HEADER_ROWS = 3


def _row_text(row):
    return ' '.join(word['text'] for word in row)


def _caption_anchors(rows):
    return [row[i] for row in rows for i in range(len(row))
            if (CAPTION_JOINED.match(row[i]['text']) or CAPTION_WORD.match(row[i]['text'])
                and i + 1 < len(row) and CAPTION_NUMBER.match(row[i + 1]['text']))
            and (i == 0 or row[i]['x0'] - row[i - 1]['x1'] > 2 * (row[i]['bottom'] - row[i]['top']))]


def _caption_span(page, rows, anchor, gutter):
    """x-range a caption's table occupies: the page, or the caption's column."""
    if gutter is None:
        return -1, page.width + 1
    near = [row for row in rows if row[0]['top'] >= anchor['top'] - 3][:HEADER_ROWS + 1]
    spans = any(word['x0'] < gutter < word['x1'] for row in near for word in row
                if word['x0'] >= anchor['x0'] - 1)
    return ((-1, page.width + 1) if spans else (-1, gutter) if anchor['x1'] <= gutter
            else (gutter, page.width + 1))


def _mostly_numeric(row):
    filled = [text for text in row[1:] if text]
    numeric = [text for text in filled
               if parse_cell(text)['kind'] in NUMERIC and not YEAR.fullmatch(text)]
    return bool(numeric) and 2 * len(numeric) >= len(filled)


def _ends_table(row, cells, gap, height):
    text = _row_text(row)
    return (gap > 2.5 * height or bool(CAPTION.match(text) or NOTE.match(text))
            or _heading_name(text) is not None
            or any(len(cell['text'].split()) >= PROSE_WORDS for cell in cells)
            or (len(cells) == 1 and (len(text.split()) > LABEL_WORDS or text.endswith('.'))))


def _caption_table(page, words, anchor, gutter, height):
    """Borderless table under a 'Table N' caption, read by word positions.

    Rows run from the first row with several cells to the row before the next
    caption, note, heading, prose line, or blank band; rows kept are the
    mostly numeric body and up to three header rows above it.
    """
    x0, x1 = _caption_span(page, word_rows(words), anchor, gutter)
    rows = word_rows([w for w in words if x0 <= w['x0'] and w['x1'] <= x1
                      and w['top'] >= anchor['top'] - 3])
    if not rows:
        return None
    cells = [_split_cells(row) for row in rows]
    first = next((i for i, row in enumerate(cells[1:], start=1) if len(row) > 1), None)
    if first is None or first > HEADER_ROWS:
        return None
    gaps = [0] + [b[0]['top'] - max(w['bottom'] for w in a) for a, b in zip(rows, rows[1:])]
    stop = next((i for i in range(first + 1, len(rows))
                 if _ends_table(rows[i], cells[i], gaps[i], height)), len(rows))
    grid = _band_rows(cells[first:stop]) if stop > first else []
    numeric = [i for i, row in enumerate(grid) if _mostly_numeric(row)]
    if not numeric or len(grid[0]) < 2:
        return None
    start = max(numeric[0] - HEADER_ROWS, 0)
    kept = grid[start:numeric[-1] + 1]
    spanned = rows[first + start:first + numeric[-1] + 1]
    after = rows[first + numeric[-1] + 1:]
    note = after[:1] if after and NOTE.match(_row_text(after[0])) else []
    return {'rows': kept, 'strategy': 'caption',
            'bbox': (max(x0, 0), spanned[0][0]['top'], min(x1, page.width),
                     max(w['bottom'] for w in spanned[-1])),
            'columns_agree': _filled(kept[0]) == max(map(_filled, kept[1:]), default=0),
            'caption': _joined(_row_text(row) for row in rows[:first]),
            'note': _joined(_row_text(row) for row in note)}


def _overlaps(box, other):
    return box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]


def _captioned(page, words, anchors, gutter, read, taken):
    found = [read(page, words, anchor, gutter, _height(words)) for anchor in anchors]
    return [table for table in found
            if table and not any(_overlaps(table['bbox'], box) for box in taken)]


IMAGE_REASON = 'table body is an image'
IMAGE_GAP = 3
LINE_GAP = 0.8
PAD = 2
SCAN_SHARE = 0.9


def _follow(rows, start, height, single=True):
    """Index after the rows set close under ``rows[start]``, each a single cell when asked."""
    close = takewhile(lambda pair: (pair[1][0]['top'] - max(w['bottom'] for w in pair[0])
                                    < LINE_GAP * height
                                    and (not single or len(_split_cells(pair[1])) == 1)),
                      zip(rows[start:], rows[start + 1:]))
    return start + 1 + sum(1 for _ in close)


def _image_table(page, words, anchor, gutter, height):
    """Region under a 'Table N' caption whose body is a picture with no text, or None.

    The body is the blank band from the caption lines to the next text row of
    the caption's column; it must be at least three text heights tall and
    overlap an embedded image. The region spans the page when the other
    column has no text beside the band.
    """
    line = [w for w in words if abs(w['top'] - anchor['top']) <= 3 and w['x0'] >= anchor['x0'] - 1]
    crosses = gutter is None or any(w['x0'] < gutter < w['x1'] for w in line)
    side = ((-1, page.width + 1) if crosses else (-1, gutter) if anchor['x1'] <= gutter
            else (gutter, page.width + 1))
    rows = word_rows([w for w in words if side[0] <= w['x0'] and w['x1'] <= side[1]
                      and w['top'] >= anchor['top'] - 3])
    end = _follow(rows, 0, height)
    floor = max(w['bottom'] for row in rows[:end] for w in row)
    stop = rows[end][0]['top'] if end < len(rows) else page.height
    beside = [w for w in words if not side[0] <= w['x0'] <= side[1]
              and floor < (w['top'] + w['bottom']) / 2 < stop]
    span = side if beside else (-1, page.width + 1)
    margins = (min(w['x0'] for w in words), max(w['x1'] for w in words))
    pictures = [(image['x0'], image['top'], image['x1'], image['bottom']) for image in page.images]
    pictured = [box for box in pictures if _overlaps((span[0], floor, span[1], stop), box)]
    # a scanned page is one picture of the whole sheet, so only smaller pictures widen the region
    inner = [box for box in pictured if box[2] - box[0] < SCAN_SHARE * page.width]
    x0 = max(span[0], min([margins[0]] + [box[0] for box in inner]))
    x1 = min(span[1], max([margins[1]] + [box[2] for box in inner]))
    if stop - floor < IMAGE_GAP * height or not pictured:
        return None
    noted = end < len(rows) and NOTE.match(_row_text(rows[end]))
    note = rows[end:_follow(rows, end, height, single=False)] if noted else []
    return {'rows': [], 'strategy': 'image', 'columns_agree': None, 'reason': IMAGE_REASON,
            'bbox': (max(x0 - PAD, 0), max(anchor['top'] - PAD, 0), min(x1 + PAD, page.width),
                     min(stop - PAD / 2, page.height)),
            'caption': _joined(_row_text(row) for row in rows[:end]),
            'note': _joined(_row_text(row) for row in note)}


def _pdf_page_tables(page, words, gutter):
    found = sorted(_page_tables(page, words), key=lambda table: table['bbox'][1])
    lines = sorted(page.extract_text_lines(x_tolerance=X_TOLERANCE) if found else [],
                   key=lambda line: line['top'])
    edges = [0] + [table['bbox'][3] for table in found]
    tops = [table['bbox'][1] for table in found[1:]] + [float('inf')]
    ruled = [{'rows': table['rows'], 'strategy': table['strategy'],
              'columns_agree': table['columns_agree'], 'bbox': table['bbox'],
              **_around(_between(lines, edges[i], table['bbox'][1]),
                        _between(lines, table['bbox'][3], tops[i]))}
             for i, table in enumerate(found)]
    anchors = _caption_anchors(word_rows(words))
    captioned = _captioned(page, words, anchors, gutter, _caption_table,
                           [table['bbox'] for table in found])
    pictured = _captioned(page, words, anchors if page.images else [], gutter, _image_table,
                          [table['bbox'] for table in ruled + captioned])
    return sorted(ruled + captioned + pictured, key=lambda table: table['bbox'][1])


def _read_page(number, page, text):
    words = page.extract_words(x_tolerance=X_TOLERANCE)
    gutter = find_gutter(page, words)
    return {'page': number, 'plain': page.extract_text(x_tolerance=X_TOLERANCE) or '',
            'tables': _pdf_page_tables(page, words, gutter),
            **(page_text(page, words, gutter) if text else {})}


def _read_pages(path, max_pages, text):
    try:
        import pdfplumber
    except ImportError:
        raise ImportError('Table reading requires pdfplumber; reinstall pubextractor') from None
    with pdfplumber.open(str(path)) as document:
        return [_read_page(number, page, text)
                for number, page in enumerate(document.pages[:max_pages or None], start=1)]


def _numbered(pages):
    labels = {p['page']: p['printed_page'] for p in detect_page_labels(
        [{'page': p['page'], 'text': p['plain']} for p in pages])}
    found = [(p['page'], table) for p in pages for table in p['tables']]
    return [{'pdf_page': page, 'printed_page': labels.get(page), 'table': number,
             'caption': table['caption'], 'note': table['note'],
             'strategy': table['strategy'], 'columns_agree': table['columns_agree'],
             'bbox': tuple(table['bbox']), 'reason': table.get('reason', ''),
             'rows': [[(cell or '').strip() for cell in row] for row in table['rows']]}
            for number, (page, table) in enumerate(found, start=1)]


def read_pdf_tables(path, max_pages=None):
    """Tables in a PDF, numbered across the document, with file and printed page.

    Each table keeps the 'Table N' caption above it and the 'Note' lines below
    it on the same page, when present. A table with no ruling is read under its
    'Table N' caption, within the caption's column on a two-column page, with
    strategy ``caption``.
    """
    return _numbered(_read_pages(path, max_pages, False))


def _paragraphs(blocks):
    return [b['text'] for b in blocks if b['kind'] == 'paragraph' and b['text'].strip()]


def _docx_tables(blocks):
    """Table blocks with the 'Table N' caption before and the 'Note' paragraph after."""
    places = [i for i, block in enumerate(blocks) if block['kind'] == 'table']
    bounds = zip([-1] + places, places, places[1:] + [len(blocks)])
    return [{**blocks[here], **_around(_paragraphs(blocks[before + 1:here]),
                                       _paragraphs(blocks[here + 1:after])[:1])}
            for before, here, after in bounds]


def table_grid(table):
    """One cell grid for a ``read_pdf_tables`` table or a ``read_docx`` table block.

    Rows are padded to one width, whitespace is collapsed, and each cell keeps
    its 1-based ``row`` and ``col`` with the table's number, location,
    caption, and note.
    """
    rows = [[' '.join((cell or '').split()) for cell in row] for row in table.get('rows') or []]
    width = max((len(row) for row in rows), default=0)
    rows = [row + [''] * (width - len(row)) for row in rows]
    return {'table': table['table'], 'pdf_page': table.get('pdf_page'),
            'printed_page': table.get('printed_page'), 'section': table.get('section', ''),
            'caption': table.get('caption', ''), 'note': table.get('note', ''), 'rows': rows,
            'strategy': table.get('strategy', ''), 'columns_agree': table.get('columns_agree'),
            'bbox': table.get('bbox'), 'reason': table.get('reason', ''),
            'cells': [{'row': r, 'col': c, 'text': text}
                      for r, row in enumerate(rows, start=1)
                      for c, text in enumerate(row, start=1)]}


IMAGE_FILES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def open_image_file(path):
    """An image file as RGB, upright by its EXIF orientation, transparency on white, 16-bit scaled.

    A multi-page TIFF gives its first page.
    """
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        picture = ImageOps.exif_transpose(source)
        if picture.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            import numpy as np
            values = np.asarray(picture, dtype=float)
            top = values.max() or 1
            picture = Image.fromarray((values * 255 / top).astype('uint8'))
        if picture.mode in ('RGBA', 'LA') or 'transparency' in picture.info:
            rgba = picture.convert('RGBA')
            picture = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba)
        return picture.convert('RGB')


def _image_file_grid(path):
    """An image file read as one table image, for a coder or a vision model to read."""
    width, height = open_image_file(path).size
    return table_grid({'table': 1, 'pdf_page': 1, 'strategy': 'image', 'reason': IMAGE_REASON,
                       'bbox': (0, 0, width, height), 'caption': Path(path).stem})


def read_tables(path, max_pages=None):
    """Cell grids for every table in a PDF, DOCX, or image file, with caption and note."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_FILES:
        return [_image_file_grid(path)]
    if suffix == '.docx':
        from .docx_reader import read_docx
        return [table_grid(table) for table in _docx_tables(read_docx(path))]
    if suffix == '.pdf':
        return [table_grid(table) for table in read_pdf_tables(path, max_pages)]
    raise ValueError('tables are read from .pdf, .docx, or image files')


def read_document(path, max_pages=None):
    """Pages as ``read_pdf`` or ``read_docx`` gives them and grids as ``read_tables`` gives them.

    A PDF is opened once and each page's words are read once for text, columns,
    and tables. An image file is one page with no text and one table image.
    """
    if Path(path).suffix.lower() in IMAGE_FILES:
        return [{'page': 1, 'text': '', 'columns': 1}], [_image_file_grid(path)]
    if Path(path).suffix.lower() == '.docx':
        from .docx_reader import read_docx
        blocks = read_docx(path)
        return blocks, [table_grid(table) for table in _docx_tables(blocks)]
    if Path(path).suffix.lower() != '.pdf':
        raise ValueError('documents are read from .pdf, .docx, or image files')
    if max_pages is not None and max_pages < 1:
        raise ValueError(f"max_pages must be a positive integer or None, got {max_pages!r}")
    pages = _read_pages(path, max_pages, True)
    texts = [{key: page[key] for key in ('page', 'text', 'columns')} for page in pages]
    report_empty(texts, path)
    return texts, [table_grid(table) for table in _numbered(pages)]


def table_pages(grids):
    """Tables as page-like blocks, so ``extract_values`` and ``extract_with_rag`` search them."""
    return [{'page': grid['pdf_page'], 'printed_page': grid['printed_page'],
             'table': grid['table'], 'section': grid['section'],
             'text': '\n'.join([grid['caption']] + [' | '.join(row) for row in grid['rows']]
                               + [grid['note']]).strip()}
            for grid in grids]


NUMERIC = {'number', 'percent', 'mean_sd', 'n_percent'}
YEAR = re.compile(r'(?:19|20)\d{2}')


def _numeric_row(row):
    return any(parse_cell(text)['kind'] in NUMERIC and not YEAR.fullmatch(text)
               for text in row[1:])


def _fill_right(row):
    return [row[0]] + list(accumulate(row[1:], lambda left, text: text or left))


def _column(col, texts):
    labeled = [header_n(text) for text in texts]
    parts = [label for label, _ in labeled]
    kept = [part for i, part in enumerate(parts) if part and part not in parts[:i]]
    stated = [(i + 1, n) for i, (_, n) in enumerate(labeled) if n is not None]
    return {'col': col, 'label': ' '.join(kept), 'parts': parts,
            'n': stated[-1][1] if stated else None, 'n_row': stated[-1][0] if stated else None}


def read_header(grid, header_rows=None):
    """Header rows above the numeric body, joined into one label per column.

    Header rows are the leading rows with no number outside the first column;
    a year such as 2019 does not count as a number here. A table with no
    numeric row has one header row. Rows holding one label just above the
    body are outcome sub-headers, not header rows. Blank cells in header rows above the last
    take the label to their left, since a merged group label is read once.
    ``header_rows`` overrides the count.
    """
    rows = grid['rows']
    detected = next((i for i, row in enumerate(rows) if _numeric_row(row)), min(1, len(rows)))
    lone = sum(1 for _ in takewhile(lone_label, reversed(rows[1:detected])))
    count = detected - lone if header_rows is None else header_rows
    stacked = [_fill_right(row) for row in rows[:max(count - 1, 0)]] + rows[max(count - 1, 0):count]
    width = len(rows[0]) if rows else 0
    return {'header_rows': count,
            'columns': [_column(c, [row[c - 1] for row in stacked])
                        for c in range(1, width + 1)]}


PAIRS = {('mean_sd', 'mean_sd'): ('mean', 'sd'), ('mean_sd', 'mean'): ('mean', 'sd'),
         ('mean_sd', None): ('mean', 'sd'), ('mean_sd', 'adj_mean_se'): ('adj_mean', 'se'),
         ('n_percent', 'n'): ('n', 'percent'), ('n_percent', 'percent'): ('n', 'percent'),
         ('n_percent', None): ('n', 'percent')}
EXPANDED = {'mean_sd': ('mean', 'sd'), 'adj_mean_se': ('adj_mean', 'se')}
SINGLE = tuple(role for role in STAT_ROLES if role not in ('mean_sd', 'adj_mean_se'))
FITS = {'number': SINGLE, 'percent': ('percent',)}
CONVENTIONS = {'mean_sd': r'\bm(?:eans?)?\s*(?:\(\s*sds?\s*\)|\u00b1\s*sds?\b)',
               'n_percent': r'\bn\s*\(\s*%\s*\)'}


def _slug(text):
    return re.sub(r'[^0-9a-z]+', '_', str(text).lower()).strip('_')


def _layer(*sources):
    return next((layer for layer in ('human', 'prompt') if layer in sources), 'table')


def _record(grid, paper_id, place, stat, value, status):
    return {'paper_id': paper_id, 'field': place.get('field', 'unmapped'),
            'value': value, 'layer': place.get('layer', 'table'), 'status': status,
            'quote': place.get('quote', ''), 'pdf_page': grid.get('pdf_page'),
            'printed_page': grid.get('printed_page'), 'paragraph': None,
            'table': grid.get('table'), 'row': place.get('row'), 'col': place.get('col'),
            'section': grid.get('section', ''), 'group': place.get('group'),
            'arm': place.get('arm'), 'time': place.get('time'), 'stat': stat,
            'outcome': place.get('outcome'),
            'candidates': place.get('candidates', 1)}


def _typed(stat, value):
    return int(value) if stat == 'n' and value is not None and float(value).is_integer() else value


def _cell_values(stat, cell, shape):
    kind = cell['kind']
    if (kind, stat) in PAIRS and (stat is not None or shape):
        return list(zip(PAIRS[(kind, stat)], (cell['value'], cell['second']))), 'fits'
    if kind == 'empty':
        return ([(role, None) for role in EXPANDED.get(stat, (stat,))], 'missing') if stat else (
            [], 'skip')
    if stat is None:
        return ([(None, cell['value'])], 'unmapped') if kind in NUMERIC else (
            [(None, None)], 'missing' if kind == 'missing' else 'text')
    if stat in FITS.get(kind, ()):
        return [(stat, cell['value'])], 'fits'
    return [(stat, None)], 'missing' if kind == 'missing' else 'mismatch'


def _namer(arms, outcomes):
    """Field names that keep arms sharing a role, and several outcomes, apart."""
    labels = {(arm['group'], arm['label']) for arm in arms}
    shared = {group for group, count in Counter(g for g, _ in labels).items() if count > 1}
    several = len({o for o in outcomes if o}) > 1

    def name(outcome, arm, time, stat):
        arm_part = (arm['group'] if arm and arm['group'] not in (None, 'other', *shared)
                    else arm['label'] if arm else None)
        parts = (outcome if several else None, arm_part, time, stat)
        return '_'.join(_slug(part) for part in parts if part) if stat else 'unmapped'
    return name


def _fill_down(texts):
    return list(accumulate(texts, lambda left, text: text or left))


def _row_outcomes(grid, header, layout, subheaders):
    col = layout['outcome_col']
    rows = range(header['header_rows'] + 1, len(grid['rows']) + 1)
    texts = [header_n(grid['rows'][r - 1][col - 1])[0] if col and r not in subheaders else ''
             for r in rows]
    names = _fill_down([label_residue(text) for text in texts])
    above = _fill_down([subheaders.get(r, '') for r in rows])
    return {r: (' '.join(filter(None, (sub, name))) or None, match_rule(TIME_RULES, text))
            for r, sub, name, text in zip(rows, above, names, texts)}


def _row_stats(grid, header, layout, columns, rows):
    """Stat of each body row when row labels name stats and column headers name arms."""
    found = {r: None if re.search(VETO, ' '.join(grid['rows'][r - 1][0].lower().split()))
             else match_rule(STAT_RULES, grid['rows'][r - 1][0]) for r in rows}
    labeled = [c for c in columns if c['kind'] is None and c['source'] != 'veto']
    transposed = (not layout['rows_are_arms'] and labeled
                  and all(c['stat'] is None for c in labeled)
                  and sum(stat is not None for stat in found.values()) >= 2)
    return found if transposed else {}


def _cell_records(grid, paper_id, column, arm, row, context):
    cell = parse_cell(grid['rows'][row - 1][column['col'] - 1])
    values, fit = _cell_values(column['stat'], cell, column['source'] != 'veto')
    outcome, row_time = context['outcomes'].get(row, (None, None))
    outcome = outcome or context['column_outcomes'].get(column['col'])
    time = column['time'] or row_time
    label = grid['rows'][row - 1][0]
    text = grid['rows'][row - 1][column['col'] - 1]
    flag = next(((e['source'], e['quote']) for e in (column, arm or {})
                 if e.get('source') in ('refused', 'failed')), None)
    cited = text if fit == 'text' else f"{label} | {column['label']}: {text}"
    quote = flag[1] if flag else cited
    place = {'row': row, 'col': column['col'], 'quote': quote,
             'group': (arm or {}).get('group'), 'arm': (arm or {}).get('label'),
             'time': time, 'outcome': outcome,
             'layer': _layer(column['source'], (arm or {}).get('source'))}
    shaped = column['stat'] is None and cell['kind'] not in context['stated']
    status = {'fits': 'review' if shaped else 'extracted', 'unmapped': 'review',
              'missing': 'not_found', 'mismatch': 'review', 'text': 'review', 'skip': None}
    if flag:
        status = dict.fromkeys(status, 'refused' if flag[0] == 'refused' else 'review')
    elif grid.get('strategy', 'lines') != 'lines' and grid.get('columns_agree') is False:
        status = {fit: 'review' if value == 'extracted' else value
                  for fit, value in status.items()}
    return [_record(grid, paper_id, {**place, 'field': context['name'](outcome, arm, time, stat)},
                    stat, _typed(stat, value), status[fit])
            for stat, value in values]


def _n_records(grid, paper_id, arms, name):
    stated = [a for a in arms if a['n'] is not None]
    firsts = [a for i, a in enumerate(stated)
              if (a['label'], a['n']) not in {(b['label'], b['n']) for b in stated[:i]}]
    places = [(a, {'row': a['n_row'], 'col': a['index'] if a['axis'] == 'col' else 1,
                   'group': a['group'], 'arm': a['label'], 'layer': _layer(a['source']),
                   'field': name(None, a, None, 'n')}) for a in firsts]
    return [_record(grid, paper_id,
                    {**place, 'quote': grid['rows'][place['row'] - 1][place['col'] - 1]},
                    'n', a['n'], 'extracted')
            for a, place in places]


def _conventions(grid, header, note):
    lines = ([' '.join(row) for row in grid['rows'][:header['header_rows']]]
             + [grid['caption'], grid['note'] if note is None else note])
    text = ' '.join(' '.join(lines).lower().split())
    return {kind for kind, pattern in CONVENTIONS.items() if re.search(pattern, text)}


def _empty_table(grid, paper_id):
    numbers = any(parse_cell(cell['text'])['kind'] in NUMERIC for cell in grid['cells'])
    reason = ('numbers found but none read as table values' if numbers
              else 'table has no numeric cells')
    place = {'field': 'table', 'quote': reason, 'candidates': 0}
    return [_record(grid, paper_id, place, None, None,
                    'review' if numbers else 'not_found')]


def _image_record(grid, paper_id):
    page = grid.get('printed_page') or grid.get('pdf_page')
    place = {'field': 'table', 'candidates': 0,
             'quote': f"table {grid['table']} on page {page} is an image: enter values or read "
                      'with a vision model'}
    return [_record(grid, paper_id, place, None, None, 'not_found')]


def _one_table(grid, paper_id, settings, model):
    if grid.get('reason'):
        return _image_record(grid, paper_id)
    header_rows, header_overrides, group_overrides, note = settings
    header = read_header(grid, header_rows)
    columns = map_headers(header, header_overrides, **model)
    arms = find_groups(grid, header, group_overrides, **model)
    layout = read_layout(grid, header)
    subheaders = subheader_rows(grid, header)
    body = [r for r in range(header['header_rows'] + 1, len(grid['rows']) + 1)
            if r not in subheaders]
    row_stats = _row_stats(grid, header, layout, columns, body)
    outcomes = _row_outcomes(grid, header, layout, subheaders)
    column_outcomes = ({c['col']: label_residue(c['label']) or None for c in columns
                        if c['kind'] is None and c['source'] != 'veto'}
                       if layout['rows_are_arms'] else {})
    by_col = {a['index']: a for a in arms if a['axis'] == 'col'}
    by_row = {a['index']: a for a in arms if a['axis'] == 'row'}
    context = {'outcomes': outcomes, 'column_outcomes': column_outcomes,
               'stated': _conventions(grid, header, note),
               'name': _namer(arms, [o for o, _ in outcomes.values()]
                              + list(column_outcomes.values()))}
    cells = [record for r in body
             for column in columns if column['kind'] is None
             for record in _cell_records(
                 grid, paper_id,
                 {**column, 'stat': row_stats[r]} if row_stats and column['source'] != 'veto'
                 else column, by_col.get(column['col'], by_row.get(r)), r, context)]
    anchored = any(r['value'] is not None or r['status'] == 'not_found' for r in cells)
    cells = cells if anchored else [r for r in cells if r['stat']]
    records = _n_records(grid, paper_id, arms, context['name']) + cells
    return records or _empty_table(grid, paper_id)


def extract_table_values(grids, paper_id=None, header_rows=None, header_overrides=None,
                         group_overrides=None, notes=None, respond=None, permission=None,
                         endpoint=None, tracker=None):
    """One record per table value, in the record format of ``extract_values``.

    Records add ``row``, ``col``, ``group`` (treatment, control, other),
    ``arm`` (the arm label as written), ``time``, ``stat``, and ``outcome``.
    ``field`` joins outcome (when a table has several), arm role (or the arm
    label when several arms share a role), time, and stat, such as
    ``treatment_post_mean``. ``layer`` is ``human`` when an override settled
    the column or arm, ``prompt`` when a model did, ``table`` otherwise. Under
    a column with no stat header the cell shape gives the stats, ``extracted``
    only when the header rows, caption, or note state the convention. A dash
    or "n.r." gives ``not_found``; text gives ``review`` with no value; a
    number or text under an unmapped or vetoed header gives ``field='unmapped'``.
    A body row holding only a label, with numbers in rows below, names the
    outcome of the rows down to the next such row and gives no records.
    Columns and arms a model could not settle because permission was refused
    or missing give ``refused``, with the reason in ``quote``; a failed model
    call gives ``review`` with ``model call failed: <ExceptionType>``. Values
    from a table found only by text layout give ``review`` unless header and
    body column counts agree. A blank cell under a mapped stat gives
    ``not_found``. A table with no values, and a document with no tables, give one
    ``field='table'`` record with the reason. ``header_rows``,
    the overrides, and ``notes`` (replacing the read note) are dicts keyed by
    table number.
    """
    model = {'respond': respond, 'permission': permission, 'endpoint': endpoint,
             'tracker': tracker, 'paper_id': paper_id}
    options = (header_rows, header_overrides, group_overrides, notes)
    records = [record for grid in grids
               for record in _one_table(grid, paper_id,
                                        tuple((option or {}).get(grid['table'])
                                              for option in options), model)]
    place = {'field': 'table', 'quote': 'no tables found', 'candidates': 0}
    return records or [_record({}, paper_id, place, None, None, 'not_found')]
