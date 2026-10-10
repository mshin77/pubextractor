"""Search patterns from plain terms, and color-coded matches in a workbook."""

import os
import re
import tempfile
from pathlib import Path

IRREGULAR = {'child': 'children', 'person': 'people', 'man': 'men', 'woman': 'women',
             'datum': 'data', 'criterion': 'criteria', 'phenomenon': 'phenomena'}


def _word_pattern(word):
    body = re.escape(word.replace('*', '\x00')).replace('\x00', r'[\w\-]*')
    if not word[-1:].isalpha():
        return body
    lower = word.lower()
    if lower in IRREGULAR:
        return f'(?:{body}(?:s|es)?|{re.escape(IRREGULAR[lower])})'
    if len(word) > 1 and lower.endswith('y') and lower[-2] not in 'aeiouy*':
        return re.escape(word[:-1]) + '(?:y|ies)'
    if len(word) > 3 and lower.endswith('is'):
        return re.escape(word[:-2]) + '(?:is|es)'
    return body + '(?:s|es)?'


def pattern_from_terms(terms):
    r"""Build a search pattern from plain terms, so nobody has to write a regex.

    A plain word also finds its plural, so "middle school" finds "middle
    schools", "study" finds "studies", "analysis" finds "analyses", and a few
    irregular forms such as "child" and "children" pair up. Takes the truncation already used in a
    database search: ``dyslex*`` finds every ending, and a star anywhere in a phrase
    truncates that word, so ``applied behavior* analys*`` reaches both spellings
    and both endings. A phrase matches across a run of whitespace or a hyphen,
    so "single case design" also finds "single-case design".
    Everything else is literal, so a bracket or a full stop searches for itself
    rather than silently becoming a wildcard.

    Args:
        terms: Comma-separated string, or a sequence of terms.

    Returns:
        str: A regex pattern, or an empty string for no terms.

    Examples:
        >>> pattern_from_terms('single-case design')
        '(?:\\bsingle\\-case(?:s|es)?[\\s\\-]+design(?:s|es)?\\b)'
        >>> pattern_from_terms('study')
        '(?:\\bstud(?:y|ies)\\b)'
    """
    if isinstance(terms, str):
        terms = re.split(r'[,;\n]', terms)

    parts = []
    for term in terms:
        term = str(term).strip()
        if not term:
            continue
        body = r'[\s\-]+'.join(_word_pattern(word) for word in term.split())
        truncated = term.endswith('*')
        term = term.rstrip('*').strip()
        if not term:
            continue
        # a boundary next to a full stop or bracket can never match, so each end
        # is anchored only when the term itself ends in a word character
        head = r'\b' if term[:1].isalnum() or term[:1] == '_' else ''
        tail = r'\b' if truncated or term[-1:].isalnum() or term[-1:] == '_' else ''
        parts.append(head + body + tail)

    if not parts:
        return ''
    return '(?:' + '|'.join(parts) + ')'


def _preserve_whitespace(xlsx_path):
    """Mark cell text as space-significant after openpyxl writes it.

    openpyxl emits ``<t>`` without ``xml:space="preserve"``, so a run that opens
    or closes on a space loses it to XML whitespace normalization and the file
    reads as damaged. Marking every text element keeps the spacing a reader
    sees between a highlighted term and the words around it.
    """
    import shutil
    import zipfile

    source = zipfile.ZipFile(str(xlsx_path))
    parts = {name: source.read(name) for name in source.namelist()}
    source.close()

    rewritten = 0
    for name in list(parts):
        if not name.startswith('xl/worksheets/'):
            continue
        body = parts[name]
        patched = body.replace(b'<t>', b'<t xml:space="preserve">')
        if patched != body:
            parts[name] = patched
            rewritten += 1

    if not rewritten:
        return 0

    temporary = Path(str(xlsx_path) + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as target:
        for name, body in parts.items():
            target.writestr(name, body)
    shutil.move(str(temporary), str(xlsx_path))
    return rewritten


def highlight_evidence(xlsx_path, patterns, fields=None, colours=None,
                       default_colour='FFFF0000', key_column=None):
    """Color the matched terms inside a workbook, in place.

    Wide layout (one row per record): each text field is marked with every
    criterion's pattern, so a reader sees what any criterion matched; each
    ``{criterion}_evidence`` column is marked with only its own pattern.

    Long layout (one row per extracted value): with ``key_column`` set, each
    row's text fields are marked with only the pattern named in that row's key
    column, so a ``quote`` beside ``field = 'grade'`` shows the grade match.

    Overlapping matches keep the first one found, so a term never carries two
    colors. Re-running is safe: an already marked cell is read back as plain
    text before it is marked again.

    Args:
        xlsx_path: Workbook to mark, first sheet, header in row 1.
        patterns: ``{criterion: regex}``, the same patterns used to search.
        fields: Text columns to mark; defaults to title and abstract, or to
            ``quote`` in the long layout.
        colours: ``{criterion: 'AARRGGBB'}``; criteria without one take
            ``default_colour``.
        default_colour: Color for any criterion missing from ``colours``.
        key_column: Column naming the criterion of each row, for the long layout.

    Returns:
        int: Number of cells marked.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.cell.rich_text import CellRichText, TextBlock
        from openpyxl.cell.text import InlineFont
    except ImportError:
        raise ImportError('Workbook highlighting requires openpyxl; reinstall pubextractor') from None

    fields = fields or (['quote'] if key_column else ['title', 'abstract'])
    names = [str(name).strip() for name in patterns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Criterion names repeat once trimmed: {repeated}")
    colors = {str(name).strip(): color for name, color in (colours or {}).items()}

    per_criterion = {name: (InlineFont(b=True, color=colors.get(name, default_colour)),
                            re.compile(pattern, re.IGNORECASE))
                     for name, pattern in zip(names, patterns.values())
                     if pattern and str(pattern).strip()}
    every = list(per_criterion.values())

    book = load_workbook(str(xlsx_path), rich_text=True)
    sheet = book.active
    header = {str(sheet.cell(1, column).value).strip(): column
              for column in range(1, sheet.max_column + 1)
              if sheet.cell(1, column).value is not None}

    targets = {f: every for f in fields if f in header}
    targets.update({f'{name}_evidence': [per_criterion[name]]
                    for name in per_criterion if f'{name}_evidence' in header})

    keyed = key_column is not None
    if keyed and key_column not in header:
        raise KeyError(f"key_column {key_column!r} is not in the header row")

    marked = 0
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        key = str(row[header[key_column] - 1].value).strip() if keyed else None
        row_targets = ({f: [per_criterion[key]] for f in fields if f in header}
                       if keyed and key in per_criterion
                       else {} if keyed else targets)
        for field_name, compiled in row_targets.items():
            cell = row[header[field_name] - 1]
            text = cell.value
            if type(text).__name__ == 'CellRichText':
                text = ''.join(str(part) for part in text)
            if not isinstance(text, str):
                continue

            taken, spans = [], []
            for color, pattern in compiled:
                for match in pattern.finditer(text):
                    if match.end() == match.start() or any(match.start() < end and match.end() > start
                           for start, end in taken):
                        continue
                    taken.append((match.start(), match.end()))
                    spans.append((match.start(), match.end(), color))
            if not spans:
                continue
            spans.sort()

            parts, last = [], 0
            for start, end, color in spans:
                if start > last:
                    parts.append(text[last:start])
                parts.append(TextBlock(color, text[start:end]))
                last = end
            if last < len(text):
                parts.append(text[last:])
            cell.value = CellRichText(*parts)
            marked += 1

    handle, staged = tempfile.mkstemp(suffix='.xlsx', dir=str(Path(xlsx_path).parent))
    os.close(handle)
    try:
        book.save(staged)
        os.replace(staged, str(xlsx_path))
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    _preserve_whitespace(xlsx_path)
    return marked
