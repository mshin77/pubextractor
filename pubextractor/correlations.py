"""Correlation matrices read from tables, checked, mapped to constructs, and written as CSV matrices."""

import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .cells import N_LABEL, parse_cell
from .output import write_csv

NUMBERED = re.compile(r'^\d{1,2}\s*[.)]\s*')
LABEL_MARKER = re.compile(r'\s*(?:\*{1,3}|[†‡§]+)$')
INDEX_HEADER = re.compile(r'\s*\d{1,2}\s*[.)]?\s*')

CORRELATION_COLUMNS = ('paper_id', 'var_i', 'var_j', 'r', 'n', 'table', 'pdf_page', 'status',
                       'checks')


def _value(text):
    cell = parse_cell(text)
    return cell['value'] if cell['kind'] == 'number' else np.nan


def correlation_checks(matrix, layout, n):
    """Flags for a correlation matrix; nothing is repaired."""
    off = matrix[~np.eye(len(matrix), dtype=bool)]
    blank = bool(np.isnan(off).any())
    flags = {'fewer than 2 variables': len(matrix) < 2,
             'diagonal not 1': not np.allclose(np.diag(matrix), 1),
             'not symmetric': layout == 'full' and not np.allclose(matrix, matrix.T,
                                                                   equal_nan=True),
             '|r| > 1': bool((np.abs(off[~np.isnan(off)]) > 1).any()),
             'blank cells': blank,
             'not positive definite': (len(matrix) >= 2 and not blank
                                       and np.linalg.eigvalsh(matrix).min() <= 0),
             'n missing': n is None}
    return [name for name, failed in flags.items() if failed]


def correlation_matrix(grid, n=None):
    """Labels, square matrix, n, and location of a correlation table grid.

    Row labels name the variables ('1. Reading' loses its number); the first
    row is the header, and the columns after the label column hold the matrix
    in row order. When the header numbers its columns (1, 2, 3), only those
    columns are read, so M, SD, or alpha columns never enter the matrix, and
    an APA lower triangle without its last column is still square. A lower,
    upper, or full triangle is read; a dash or blank diagonal is 1, and the
    missing triangle mirrors the other. Footnote markers leave the labels.
    Fewer filled columns than labels adds ``not_square``, and no cell is
    filled in. n comes from 'N = 120' in the caption or note, else from ``n``.
    """
    header = grid['rows'][0]
    body = [row for row in grid['rows'][1:] if row[0].strip()]
    labels = [LABEL_MARKER.sub('', NUMBERED.sub('', row[0])) for row in body]
    k = len(labels)
    indexed = [c for c in range(1, len(header)) if INDEX_HEADER.fullmatch(header[c] or '')][:k]
    filled = sum(any(row[c].strip() for row in grid['rows'][:1] + body)
                 for c in range(1, len(header)))
    square = len(indexed) >= k - 1 if indexed else filled >= k
    columns = indexed or list(range(1, k + 1))
    matrix = np.array([[_value(row[c] if c < len(row) else '') for c in columns]
                       + [np.nan] * (k - len(columns)) for row in body],
                      dtype=float).reshape(k, k)
    diagonal = np.diag(matrix)
    matrix[np.diag_indices(k)] = np.where(np.isnan(diagonal) & square, 1.0, diagonal)
    lower = ~np.isnan(matrix[np.tril_indices(k, -1)]).all()
    upper = ~np.isnan(matrix[np.triu_indices(k, 1)]).all()
    layout = 'full' if lower and upper else 'upper' if upper else 'lower'
    matrix = np.where(np.isnan(matrix) & square, matrix.T, matrix)
    stated = N_LABEL.search(f"{grid.get('caption', '')} {grid.get('note', '')}")
    found = int(stated.group(1).replace(',', '')) if stated else n
    return {'labels': labels, 'matrix': matrix, 'layout': layout, 'n': found,
            'n_source': 'table' if stated else 'study' if n is not None else None,
            'table': grid['table'], 'pdf_page': grid.get('pdf_page'),
            'printed_page': grid.get('printed_page'), 'section': grid.get('section', ''),
            'checks': ['not_square'] * (not square) + correlation_checks(matrix, layout, found)}


def _record(result, paper_id, i=None, j=None):
    r = None if i is None or np.isnan(result['matrix'][j, i]) else float(result['matrix'][j, i])
    return {'paper_id': paper_id, 'var_i': None if i is None else result['labels'][i],
            'var_j': None if j is None else result['labels'][j], 'r': r, 'n': result['n'],
            'table': result['table'], 'pdf_page': result['pdf_page'],
            'status': 'not_found' if r is None else 'review' if result['checks'] else 'extracted',
            'checks': '; '.join(result['checks'])}


def correlation_records(results, paper_id=None):
    """One record per variable pair in ``CORRELATION_COLUMNS`` order, for ``write_csv``.

    A matrix with fewer than two variables gives one ``not_found`` record.
    """
    return [record for result in results
            for record in ([_record(result, paper_id, i, j)
                            for i, j in zip(*np.triu_indices(len(result['labels']), 1))]
                           or [_record(result, paper_id)])]


def _normal(text):
    return ' '.join(re.sub(r'[^0-9a-z]+', ' ', str(text).lower()).split())


def map_constructs(labels, constructs, overrides=None):
    """Construct for each label by exact, then normalized, match; ``overrides`` always win.

    A construct matched exactly is not matched again; one matched by two
    normalized labels is left unmapped for both. Unmapped labels are listed,
    never guessed. ``overrides`` maps a label to a construct or ``None``.
    """
    overrides = overrides or {}
    problems = ([f'{label!r} is not a label' for label in overrides if label not in labels]
                + [f'{c!r} is not a construct' for c in overrides.values()
                   if c is not None and c not in constructs])
    if problems:
        raise ValueError('Override problems: ' + '; '.join(problems))
    counts = Counter(_normal(c) for c in constructs)
    by_name = {_normal(c): c for c in constructs if counts[_normal(c)] == 1}
    exact = {label: label for label in labels if label in constructs}
    loose = {label: by_name.get(_normal(label)) for label in labels if label not in exact}
    taken = Counter(loose.values())
    mapping = {**{label: c if taken[c] == 1 and c not in exact else None
                  for label, c in loose.items()}, **exact, **overrides}
    return {'mapping': {label: c for label, c in mapping.items() if c},
            'unmapped': [label for label in labels if not mapping[label]],
            'sources': {label: 'human' if label in overrides else 'exact'
                        if label in constructs else 'normalized'
                        for label in labels if mapping[label]}}


SIMILAR = 0.8


def suggest_constructs(label_sets):
    """Construct names suggested from the labels of several studies, most shared first.

    Labels that match after case and punctuation are one construct, named by
    its first spelling; constructs are ordered by how many studies use them,
    and a name at least ``SIMILAR`` like one already kept is folded into it.
    """
    first = {}
    [first.setdefault(_normal(label), label) for labels in label_sets for label in labels]
    shared = Counter(name for labels in label_sets for name in {_normal(label) for label in labels})
    order = list(first)
    ranked = sorted(order, key=lambda name: (-shared[name], order.index(name)))
    kept = []
    [kept.append(name) for name in ranked
     if all(SequenceMatcher(None, name, other).ratio() < SIMILAR for other in kept)]
    return [first[name] for name in kept]


def suggest_mapping(labels, constructs, overrides=None, threshold=SIMILAR):
    """A construct and its reason for each label, for a coder to confirm or change.

    Exact, then normalized matches as ``map_constructs`` gives them; a label
    still open takes the one free construct whose name is at least
    ``threshold`` similar (difflib ratio after normalizing), else none.
    """
    found = map_constructs(labels, constructs, overrides)
    used = set(found['mapping'].values())
    scores = {label: sorted(((SequenceMatcher(None, _normal(label), _normal(c)).ratio(), c)
                             for c in constructs if c not in used), reverse=True)
              for label in found['unmapped']}
    close = {label: ranked[0] for label, ranked in scores.items()
             if ranked and ranked[0][0] >= threshold
             and (len(ranked) == 1 or ranked[1][0] < threshold)}
    taken = Counter(c for _, c in close.values())
    return {label: (found['mapping'][label], found['sources'][label]) if label in found['mapping']
            else (close[label][1], f"similar {close[label][0]:.2f}")
            if label in close and taken[close[label][1]] == 1 else (None, 'no match')
            for label in labels}


def _shared(study, constructs):
    index = {label: i for i, label in enumerate(study['labels'])}
    where = {c: index[label] for label, c in study['mapping'].items()}
    matrix = np.asarray(study['matrix'], dtype=float)
    return np.array([[matrix[where[a], where[b]] if a in where and b in where else np.nan
                      for b in constructs] for a in constructs])


def to_matrices(studies, constructs, folder):
    """Per-study CSV matrices over ``constructs`` and ``n.csv``.

    Each study is a dict with ``study``, ``labels``, ``matrix``, ``n``, and
    ``mapping`` (label to construct, from ``map_constructs``). Constructs a
    study did not measure are ``NA``. A matrix whose diagonal is not 1 or
    with |r| > 1 is refused.
    """
    if 'construct' in constructs or len(set(constructs)) != len(constructs):
        raise ValueError('Export problems: construct columns repeat or use the reserved construct column')
    problems = ([f"{s['study']}: n missing" for s in studies if s.get('n') is None]
                + [f"{s['study']}: not a correlation matrix ({', '.join(bad)})" for s in studies
                   for bad in [[c for c in correlation_checks(np.asarray(s['matrix'], dtype=float),
                                                              'full', s.get('n'))
                                if c in ('diagonal not 1', '|r| > 1', 'not symmetric')]] if bad]
                + [f"{s['study']}: two labels map to one construct" for s in studies
                   if len(set(s['mapping'].values())) < len(s['mapping'])])
    if problems:
        raise ValueError('Export problems: ' + '; '.join(problems))
    files = [re.sub(r'[^\w.-]', '_', str(s['study'])) for s in studies]
    names = [file.casefold().rstrip(' .') for file in files] + ['n']
    if len(set(names)) != len(names):
        raise ValueError('Export problems: matrix filenames collide with another study or n.csv')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for study, file in zip(studies, files):
        write_csv([{'construct': a, **{b: 'NA' if np.isnan(r) else f'{r:.6g}'
                                       for b, r in zip(constructs, row)}}
                   for a, row in zip(constructs, _shared(study, constructs))],
                  folder / f'{file}.csv')
    write_csv([{'study': s['study'], 'file': file, 'n': s['n']}
               for s, file in zip(studies, files)], folder / 'n.csv')
    return [folder / f'{file}.csv' for file in files] + [folder / 'n.csv']
