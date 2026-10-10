import logging
import re
from collections import Counter
from itertools import accumulate, groupby

import numpy as np

logger = logging.getLogger(__name__)

SECTION_NAMES = (
    'abstract', 'introduction', 'background', 'literature review', 'method',
    'methods', 'methodology', 'materials and methods', 'participants',
    'measures', 'procedure', 'analysis', 'data analysis', 'results', 'findings', 'discussion',
    'limitations', 'conclusion', 'conclusions', 'references', 'appendix',
)

_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+')
_ABBREVIATION_END = re.compile(r'(?i)(?:\bet al|\be\.g|\bi\.e|\bFig|\bvs)\.$')
_HEADING_MAX_WORDS = 6

_HEADING_START = re.compile(
    r'(?i)(?:' + '|'.join(sorted(SECTION_NAMES, key=len, reverse=True)) + r')\b')
_NUMBERING = re.compile(r'(?:\d+(?:\.\d+)*\.?|[IVX]+\.|[A-Z]\.)\s+')
_TRAILING_NOISE = re.compile(r'(?:\s+[^\w\s]+)+$')
_FUNCTION_WORDS = frozenset(('a', 'an', 'and', 'of', 'the', 'in', 'for', 'to', 'on', 'with', 'or'))

_GUTTER_ZONE = (0.35, 0.65)
_MIN_SPLIT_ROWS = 5
_MIN_TABLE_ROWS = 3

_PAGE_OF = re.compile(r'(?i)\bpage\s+(\d{1,4})\s+of\s+(\d{1,4})\b')
_BARE_NUMBER = re.compile(r'\d{1,4}')

# OCR layers set word gaps under pdfplumber's 3 pt default; tracked type stays under 1.5 pt
X_TOLERANCE = 1.5


def read_pdf(path, max_pages=None):
    """Read a born-digital PDF into per-page text.

    Requires the ``fulltext`` extra. Pages that yield no text are kept with an
    empty string so page numbers stay aligned with the document. A page split
    by a central gutter is read left column, then right column, with
    full-width rows (title, running head, wide table) kept in place; each
    page records ``columns`` as 1 or 2.
    """
    if max_pages is not None and max_pages < 1:
        raise ValueError(f"max_pages must be a positive integer or None, got {max_pages!r}")
    try:
        import pdfplumber
    except ImportError:
        raise ImportError('Full-text reading requires pdfplumber; reinstall pubextractor') from None

    with pdfplumber.open(path) as document:
        pages = [{'page': number, **page_text(page, words, find_gutter(page, words))}
                 for number, page in enumerate(document.pages[:max_pages], start=1)
                 for words in [page.extract_words(x_tolerance=X_TOLERANCE)]]
    report_empty(pages, path)
    return pages


def page_text(page, words, gutter):
    """Text and column count of one page from its words and gutter."""
    text = (page.extract_text(x_tolerance=X_TOLERANCE) if gutter is None
            else _column_text(page, gutter, words))
    return {'text': text or '', 'columns': 1 if gutter is None else 2}


def report_empty(pages, path):
    empty = sum(1 for p in pages if not p['text'].strip())
    if empty == len(pages):
        logger.warning(f"No extractable text in {path}; the file is likely scanned")
    elif empty:
        logger.info(f"{empty} of {len(pages)} pages had no extractable text")


def word_rows(words, tolerance=3):
    """Words grouped into rows by top edge, each row ordered left to right."""
    ordered = sorted(words, key=lambda word: word['top'])
    tops = list(accumulate((word['top'] for word in ordered),
                           lambda left, top: left if top - left <= tolerance else top))
    return [sorted((w for w, t in zip(ordered, tops) if t == top), key=lambda w: w['x0'])
            for top in dict.fromkeys(tops)]


def _height(words):
    return float(np.median([w['bottom'] - w['top'] for w in words]))


def _crosses(row, gutter):
    return any(w['x0'] < gutter < w['x1'] for w in row)


def find_gutter(page, words=None):
    """x of a vertical gap near the page middle that splits text into two columns, or None.

    A gap counts on a row when it is wider than a word is tall, so ordinary word
    spacing never forms a gutter. The gap must split at least five rows and more
    rows than words cross it.
    """
    words = page.extract_words(x_tolerance=X_TOLERANCE) if words is None else words
    if not words:
        return None
    rows, height = word_rows(words), _height(words)
    gaps = np.array([(a['x1'], b['x0']) for row in rows for a, b in zip(row, row[1:])
                     if b['x0'] - a['x1'] >= height]).reshape(-1, 2)
    xs = np.arange(page.width * _GUTTER_ZONE[0], page.width * _GUTTER_ZONE[1])
    counts = ((gaps[:, :1] <= xs) & (xs <= gaps[:, 1:])).sum(axis=0)
    best = int(counts.max()) if counts.size else 0
    first = int(counts.argmax()) if counts.size else 0
    tail = counts[first:] == best
    last = first + (int(tail.argmin()) if not tail.all() else tail.size) - 1
    gutter = (xs[first] + xs[last]) / 2 if counts.size else None
    split = best >= _MIN_SPLIT_ROWS and best > sum(_crosses(row, gutter) for row in rows)
    return float(gutter) if split else None


def _row_kinds(rows, gutter, height):
    """'full' for rows read across the page, 'column' for rows read per column.

    A row is full when a word crosses the gutter, when it is the first or last
    row set apart by a blank band (running head, footer), or when it sits in a
    run of three or more rows whose right part misses the right column's margin
    (a table spanning both columns); a single such row is an OCR stray.
    """
    rights = [min((w['x0'] for w in row if w['x0'] >= gutter), default=None)
              if any(w['x1'] <= gutter for w in row) else None for row in rows]
    margins = Counter(round(right) for right in rights if right is not None)
    margin = margins.most_common(1)[0][0] if margins else gutter
    off = [right is not None and not margin - height / 2 <= right <= margin + 4 * height
           for right in rights]
    spans = [flag and size >= _MIN_TABLE_ROWS for flag, group in groupby(off)
             for size in [len(list(group))] for _ in range(size)]
    tops = [min(w['top'] for w in row) for row in rows]
    bottoms = [max(w['bottom'] for w in row) for row in rows]
    apart = ([len(rows) > 1 and tops[1] - bottoms[0] > 2 * height]
             + [False] * max(len(rows) - 2, 0)
             + [len(rows) > 1 and tops[-1] - bottoms[-2] > 2 * height])[:len(rows)]
    return ['full' if _crosses(row, gutter) or edge or span else 'column'
            for row, edge, span in zip(rows, apart, spans)]


def _column_text(page, gutter, words):
    """Page text with full-width rows in place and each column block read left then right."""
    rows = word_rows(words)
    runs = [(kind, [row for _, row in group]) for kind, group in
            groupby(zip(_row_kinds(rows, gutter, _height(words)), rows), key=lambda pair: pair[0])]
    tops = [min(w['top'] for row in block for w in row) for _, block in runs]
    bottoms = [max(w['bottom'] for row in block for w in row) for _, block in runs]
    cuts = [-np.inf] + [(b + t) / 2 for b, t in zip(bottoms, tops[1:])] + [np.inf]
    sides = {'full': [(-np.inf, np.inf)], 'column': [(-np.inf, gutter), (gutter, np.inf)]}
    regions = [(x0, top, x1, bottom) for (kind, _), top, bottom in zip(runs, cuts, cuts[1:])
               for x0, x1 in sides[kind]]
    texts = [_region_text(page, region) for region in regions]
    return '\n'.join(text for text in texts if text.strip())


def _region_text(page, region):
    x0, top, x1, bottom = region

    def inside(obj):
        return obj.get('object_type') != 'char' or (
            x0 <= (obj['x0'] + obj['x1']) / 2 < x1 and top <= (obj['top'] + obj['bottom']) / 2 < bottom)
    return page.filter(inside).extract_text(x_tolerance=X_TOLERANCE) or ''


def detect_page_labels(pages):
    """Attach the page number printed in the document to each page.

    Reads ``Page N of M`` running heads and bare footer integers. A label is
    accepted only when one printed-to-file offset holds on at least two pages and
    on a majority of the pages that carry a candidate number.

    Returns:
        list: pages with ``printed_page`` set to the printed number, or to None
        throughout when no consistent numbering was found.
    """
    candidates = {}
    for index, page in enumerate(pages):
        lines = [line.strip() for line in page['text'].splitlines() if line.strip()]
        if not lines:
            continue
        for line in lines[:2] + lines[-2:]:
            explicit = _PAGE_OF.search(line)
            if explicit:
                candidates[index] = int(explicit.group(1))
                break
            if _BARE_NUMBER.fullmatch(line):
                candidates[index] = int(line)
                break

    offsets = Counter(value - index for index, value in candidates.items())
    offset, support = offsets.most_common(1)[0] if offsets else (0, 0)
    if support < 2 or support * 2 <= len(candidates):
        if candidates:
            logger.warning("Printed page numbers were inconsistent; using file positions")
        return [{**page, 'printed_page': None} for page in pages]

    return [{**page, 'printed_page': index + offset} for index, page in enumerate(pages)]


def _heading_name(line):
    """Heading text without its numbering, or None when the line is not a heading."""
    text = _TRAILING_NOISE.sub('', _NUMBERING.sub('', line.strip(), count=1))
    words = text.split()
    if not (words and len(words) <= _HEADING_MAX_WORDS and _HEADING_START.match(text)
            and not text.endswith(('.', '!', '?'))):
        return None
    if text.isupper():
        return ' '.join(w if i == 0 or w.lower() not in _FUNCTION_WORDS else w.lower()
                        for i, w in enumerate(w.capitalize() for w in words))
    cased = all(w[0].isupper() or w.lower() in _FUNCTION_WORDS for w in words)
    return text if cased else None


def detect_sections(pages):
    """Label each page with section headings and where they sit in its text.

    A heading is a short, heading-cased line (all capitals, or every word but the
    short function words capitalised) without sentence-ending punctuation that
    opens with a vocabulary word, after an optional numbering prefix. A heading
    style the vocabulary misses leaves the previous label in place.

    Returns:
        list: pages with ``section`` (last heading on the page), ``section_start``
        (heading in effect at the top of the page), and ``section_marks``
        (``(offset, heading)`` pairs, offsets into the page text).
    """
    labeled, current = [], ''
    for page in pages:
        start, marks, offset = current, [], 0
        for line in page['text'].splitlines(keepends=True):
            name = _heading_name(line)
            if name:
                current = name
                marks.append((offset, current))
            offset += len(line)
        labeled.append({**page, 'section': current, 'section_start': start,
                        'section_marks': marks})
    return labeled


def section_at(page, offset):
    """Heading in effect at a character offset of a page's text."""
    earlier = [name for start, name in page.get('section_marks', []) if start <= offset]
    return earlier[-1] if earlier else page.get('section_start', page.get('section', ''))


def split_sentences(text):
    """Yield ``(offset, sentence)``, not splitting after common abbreviations."""
    start = 0
    for gap in _SENTENCE_BREAK.finditer(text):
        if _ABBREVIATION_END.search(text[start:gap.start()]):
            continue
        yield start, text[start:gap.start()]
        start = gap.end()
    yield start, text[start:]


def similarity_scores(texts, description, model):
    """Cosine similarity of each text to a description, on the encoder's embeddings.

    A failed encoding, one row short, or non-finite values return an empty list
    rather than raising, so evidence still comes back without scores.
    """
    texts = list(texts)
    if not texts:
        return []
    try:
        matrix = np.atleast_2d(np.asarray(model.encode(texts), dtype=float))
        target = np.asarray(model.encode([description]), dtype=float).ravel()
        if matrix.shape[0] != len(texts) or not (np.isfinite(matrix).all()
                                                 and np.isfinite(target).all()):
            logger.warning("Encoder output does not match the texts; scores skipped")
            return []
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
        norms[norms == 0] = 1e-12
        return ((matrix @ target) / norms).tolist()
    except Exception:
        logger.exception("Semantic scoring failed")
        return []


def extract_fulltext_evidence(path, inclusion_patterns, model=None, max_pages=None,
                              window_words=0):
    """Locate criterion evidence in a PDF, anchored to page and section.

    A criterion with no matching span returns an empty list, not a verdict.

    Args:
        path: PDF file path.
        inclusion_patterns: ``{criterion: {'pattern': regex, 'description': text}}``,
            the structure pubmlp's ``regex_screen`` takes.
        model: Sentence encoder for similarity against the criterion description.
            Omitted, spans come back without similarity scores.
        max_pages: Stop after this many pages.
        window_words: Words of context either side of the match. Zero returns the
            whole sentence.

    Returns:
        dict: ``{criterion: [{'page', 'section', 'text', 'matched_term',
        'similarity'}]}``.
    """
    pages = detect_sections(detect_page_labels(read_pdf(path, max_pages=max_pages)))
    evidence = {name: [] for name in inclusion_patterns}

    for criterion, spec in inclusion_patterns.items():
        if 'pattern' not in spec:
            raise ValueError(f"Criterion {criterion!r} has no 'pattern' key")
        try:
            pattern = re.compile(spec['pattern'], re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"Criterion {criterion!r} has an invalid pattern "
                             f"{spec['pattern']!r}: {error}") from error
        for page in pages:
            for offset, sentence in split_sentences(page['text'].replace('\n', ' ')):
                match = pattern.search(sentence)
                if not match:
                    continue
                text = sentence.strip()
                if window_words:
                    words = text.split()
                    hit = len(text[:match.start()].split())
                    lo = max(0, hit - window_words)
                    text = ' '.join(words[lo:hit + window_words + 1])
                evidence[criterion].append({
                    'page': page.get('printed_page') or page['page'],
                    'pdf_page': page['page'],
                    'printed_page': page.get('printed_page'),
                    'section': section_at(page, offset + match.start()),
                    'text': text,
                    'matched_term': match.group(),
                    'similarity': None,
                })

    if model is not None:
        for criterion, spans in evidence.items():
            if not spans:
                continue
            scores = similarity_scores([s['text'] for s in spans],
                                       inclusion_patterns[criterion].get('description', ''),
                                       model)
            for span, score in zip(spans, scores):
                span['similarity'] = score

    found = {c: len(v) for c, v in evidence.items()}
    logger.info(f"{path}: spans per criterion {found}")
    return evidence


def format_anchor(span):
    """Citation anchor for a span, as the reviewer sees it.

    Falls back to the file position, marked as such, so a reader is never given
    a page number the document does not carry.
    """
    section = f" · {span['section']}" if span.get('section') else ''
    if span.get('printed_page'):
        return f"p. {span['printed_page']}{section}"
    if span.get('pdf_page') is not None:
        return f"file p. {span['pdf_page']}{section}"
    section_name = span.get('section') or ''
    where = [f"{label} {span[key]}" for key, label in (('paragraph', 'para.'), ('table', 'table'))
             if span.get(key)]
    return ' '.join(where + ([section_name] if section_name else [])) or 'document'
