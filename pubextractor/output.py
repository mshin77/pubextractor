"""Extracted records as long or wide rows, and as CSV."""

import csv

COLUMNS = ('paper_id', 'field', 'value', 'layer', 'status', 'quote',
           'pdf_page', 'printed_page', 'paragraph', 'table', 'row', 'col', 'section',
           'group', 'arm', 'time', 'stat', 'outcome', 'candidates')

FLAGGED = ('review', 'refused', 'not_found')


def to_long(records):
    """One row per extracted value, columns in ``COLUMNS`` order."""
    return [{column: record.get(column) for column in COLUMNS} for record in records]


def _cells(group):
    values = [str(r['value']) for r in group if r.get('value') is not None]
    flags = list(dict.fromkeys(r['status'] for r in group if r.get('status') in FLAGGED))
    return values, flags


def _column(record, split):
    table = record.get('table')
    return f"{record['field']}_t{table}" if record['field'] in split and table is not None         else record['field']


def to_wide(records):
    """One row per paper, one column per field; values joined with '; '.

    A field holding a review, refused, or not_found record gains a ``<field>_status``
    column. A field found in several tables, or in a table and in the text, gets
    one ``<field>_t<table>`` column per table so values are never joined across tables.
    """
    tables = {}
    [tables.setdefault(r['field'], set()).add(r.get('table')) for r in records]
    split = {field for field, seen in tables.items() if len(seen) > 1}
    papers = dict.fromkeys(r['paper_id'] for r in records)
    rows = {paper: {'paper_id': paper} for paper in papers}
    groups = {}
    [groups.setdefault((r['paper_id'], _column(r, split)), []).append(r) for r in records]
    for (paper, column), group in groups.items():
        values, flags = _cells(group)
        rows[paper].update({column: '; '.join(values)})
        rows[paper].update({f'{column}_status': '; '.join(flags)} if flags else {})
    return list(rows.values())


def write_csv(rows, path):
    """Write rows to a UTF-8 CSV with a header from every key seen."""
    if not rows:
        raise ValueError('no rows to write')
    header = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path
