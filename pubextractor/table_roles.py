"""Stat roles of table columns and group labels of rows and columns: rules, a model, a human."""

import json
import logging
import re
from functools import reduce
from itertools import accumulate

from .cells import header_n, parse_cell
from .confidence import quote_found
from .layers import _parse
from .licensing import redact_endpoint
from .rag import UNCHECKED, _check_permission

logger = logging.getLogger(__name__)

STAT_ROLES = ('mean', 'sd', 'mean_sd', 'n', 'se', 'adj_mean', 'adj_mean_se', 'percent',
              'r', 't', 'f', 'p')
TIMES = ('pre', 'post', 'follow_up')

ADJUSTED = r'\badj(?:usted)?\b\.?'
ADJUSTED_MEAN = ADJUSTED + r'\s*(?:(?:pre|post)(?:test)?\s+|follow[- ]?up\s+)?(?:m|means?)'

STAT_RULES = (
    ('adj_mean_se', ADJUSTED_MEAN + r'\s*\(\s*se\s*\)'),
    ('adj_mean', ADJUSTED_MEAN + r'\b|\b(?:emm|ls\s*means?|estimated marginal means?)\b'),
    ('mean_sd', r'\bm(?:ean)?\s*(?:\(\s*sd\s*\)|±\s*sd\b)'),
    ('se', r'\b(?:se|s\.e\.|sem|standard error)(?:\W|$)'),
    ('sd', r'\b(?:sd|s\.d\.|std\.?\s*dev(?:iation)?|standard deviation)(?:\W|$)'),
    ('mean', r'\b(?:m|means?|average|avg)\b'),
    ('n', r'\bn\b'),
    ('percent', r'%|\bpercent(?:age)?\b'),
    ('r', r'\br\b'),
    ('t', r'\bt\b'),
    ('f', r'\bf\b'),
    ('p', r'\bp\b'),
)

TIME_RULES = (
    ('follow_up', r'\b(?:follow[- ]?up|maintenance|delayed post(?:test)?)\b'),
    ('pre', r'\b(?:pre|pretest|baseline|time 1|t1)\b'),
    ('post', r'\b(?:post|posttest|time 2|t2)\b'),
)

# 's.d.' holds a lone d between dots, so d and g count only as bare words
VETO = (r'\b(?:diff(?:erences?)?|change|gain|ci|confidence|effect|es)\b'
        r'|(?<![.\w])[dg](?![.\w])')

LABEL_COLUMNS = {'outcome': r'(?:dependent )?(?:outcomes?|measures?|variables?)',
                 'arm': r'(?:groups?|conditions?|arms?)'}

ASK = ('Each numbered line is a {what} copied from a table. For each line choose one of '
       '{choices}, or null when none fits.\n\n{lines}\n\n'
       'Reply with JSON only: {{"labels": [{{"number": <line number>, '
       '"choice": <one of the choices or null>, "quote": "<the line copied exactly>"}}]}}')


def _squash(text):
    return ' '.join(str(text).lower().split())


def match_rule(rules, label):
    text = _squash(label)
    return next((role for role, pattern in rules if re.search(pattern, text)), None)


def label_residue(label):
    """A label without its stat and time words, such as 'Reading' from 'Reading pre M'."""
    patterns = [pattern for _, pattern in STAT_RULES + TIME_RULES] + [ADJUSTED, r'\(\s*\)']
    text = reduce(lambda left, pattern: re.sub(pattern, ' ', left, flags=re.IGNORECASE),
                  patterns, label or '')
    return ' '.join(text.split()).strip(' ,;:()-')


def _column_kind(column):
    text = _squash(column['label'])
    kind = next((kind for kind, pattern in LABEL_COLUMNS.items() if re.fullmatch(pattern, text)),
                None)
    return kind or ('stub' if column['col'] == 1 else None)


def _kept(pick, labels, choices):
    number = pick.get('number') if isinstance(pick, dict) else None
    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= len(labels):
        return False
    quote = pick.get('quote')
    return (pick.get('choice') in choices and isinstance(quote, str)
            and (_squash(quote) == _squash(labels[number - 1])
                 or quote_found(quote, [labels[number - 1]])))


def _ask_model(labels, choices, what, model):
    """Choices a model makes for labels, kept only when listed and quoting the label.

    Returns the choices and an issue: the license refusal (nothing is sent) or
    the exception name of a failed call, each with the quote for its records.
    """
    if not labels or model.get('respond') is None:
        return {}, None
    try:
        checked = _check_permission(model.get('permission'), model.get('endpoint'))
    except (PermissionError, ValueError) as refusal:
        return {}, {'source': 'refused', 'quote': str(refusal)}
    lines = '\n'.join(f'[{i}] {label}' for i, label in enumerate(labels, start=1))
    prompt = ASK.format(what=what, choices=json.dumps(list(choices)), lines=lines)
    tracker, endpoint = model.get('tracker'), model.get('endpoint')
    if tracker is not None:
        tracker.log_permission({**({'verdict': UNCHECKED} if checked is None else checked),
                                'endpoint': redact_endpoint(endpoint) if endpoint else None},
                               paper_id=model.get('paper_id'))
        tracker.log_prompt(prompt, model=getattr(model['respond'], 'model', None))
    try:
        answer = model['respond'](prompt)
    except Exception as failure:
        logger.exception('Model call failed for table labels')
        return {}, {'source': 'failed', 'quote': f'model call failed: {type(failure).__name__}'}
    picks = _parse(answer).get('labels')
    picks = picks if isinstance(picks, list) else []
    return {labels[p['number'] - 1]: p['choice'] for p in picks if _kept(p, labels, choices)}, None


def _settle(entry, role, asked, chosen, issue):
    if entry['source'] != 'rule' or entry[role] is not None:
        return entry
    if entry['label'] in chosen:
        return {**entry, role: chosen[entry['label']], 'source': 'prompt'}
    if issue and entry['label'] in asked:
        return {**entry, **issue}
    return {**entry, 'source': 'none'}


def _overrides(overrides, role, allowed, keys):
    """Overrides as dicts, checked; raises ``ValueError`` naming every problem."""
    entries = {key: value if isinstance(value, dict) else {role: value}
               for key, value in (overrides or {}).items()}
    problems = ([f'{key!r} names no column or row' for key in entries if key not in keys]
                + [f'{key!r}: {name} must be one of {list(allowed[name])} or None'
                   for key, entry in entries.items() for name, value in entry.items()
                   if name not in allowed or (value is not None and value not in allowed[name])])
    if problems:
        raise ValueError('Override problems: ' + '; '.join(problems))
    return entries


def _apply(entry, overrides, keys):
    hit = next((overrides[key] for key in keys if key in overrides), None)
    return entry if hit is None else {**entry, **hit, 'source': 'human'}


def _first_map(column):
    label, kind = column['label'], _column_kind(column)
    vetoed = kind is None and bool(re.search(VETO, _squash(label)))
    source = 'stub' if kind else 'veto' if vetoed else 'rule'
    quote = ('label names a difference, change, interval, or effect size' if vetoed else label)
    return {'col': column['col'], 'label': label, 'kind': kind,
            'stat': match_rule(STAT_RULES, label) if source == 'rule' else None,
            'time': match_rule(TIME_RULES, label), 'source': source, 'quote': quote}


def _post_default(column):
    adjusted = column['stat'] in ('adj_mean', 'adj_mean_se') and column['time'] is None
    return {**column, 'time': 'post'} if adjusted else column


def map_headers(header, overrides=None, respond=None, permission=None, endpoint=None,
                tracker=None, paper_id=None):
    """Stat role and time point of each column: rules, then a model, then ``overrides``.

    Label columns (the first column, and any headed Outcome, Measure,
    Variable, Group, Condition, or Arm) get no role and ``source='stub'``.
    Labels naming a difference, change, gain, interval, or effect size get no
    role from rules or a model (``source='veto'``). A model, when given,
    sees only the other labels the rules left unmapped and may only choose
    from ``STAT_ROLES``, quoting the label. ``overrides`` maps a column number
    or label to a role, or to ``{'stat': ..., 'time': ...}``, and always wins.
    Adjusted means without a time point are taken as posttest.
    """
    columns = [_first_map(c) for c in header['columns']]
    allowed = {'stat': STAT_ROLES, 'time': TIMES}
    keys = {k for c in columns for k in (c['col'], c['label'])}
    human = _overrides(overrides, 'stat', allowed, keys)
    asked = list(dict.fromkeys(c['label'] for c in columns
                               if c['source'] == 'rule' and c['stat'] is None and c['label']
                               and c['col'] not in human and c['label'] not in human))
    model = {'respond': respond, 'permission': permission, 'endpoint': endpoint,
             'tracker': tracker, 'paper_id': paper_id}
    chosen, issue = _ask_model(asked, STAT_ROLES, 'column header', model)
    return [_post_default(_apply(_settle(c, 'stat', asked, chosen, issue), human,
                                 (c['col'], c['label'])))
            for c in columns]


GROUP_ROLES = ('treatment', 'control', 'other')

GROUP_RULES = (
    ('control', r'\b(?:controls?|comparisons?|business[- ]as[- ]usual|bau'
                r'|wait(?:ing)?[- ]?list(?:ed)?|treatment[- ]as[- ]usual|untreated|placebo'
                r'|no[- ](?:intervention|treatment|training)|non[- ]?intervention|status quo'
                r'|(?:typical|usual|standard|regular|traditional) (?:practice|instruction|care'
                r'|curriculum)|delayed (?:intervention|treatment))\b'),
    ('treatment', r'\b(?:intervention|experimental|treat(?:ment|ed))\b'),
)


BLANK = ('empty', 'missing')


def lone_label(row):
    """The one text cell of a row whose other cells are blank or dashes, or None."""
    kept = [text for text in row if parse_cell(text)['kind'] not in BLANK]
    return kept[0] if len(kept) == 1 and parse_cell(kept[0])['kind'] == 'text' else None


def _has_number(row):
    return any(parse_cell(text)['kind'] not in BLANK + ('text',) for text in row[1:])


def subheader_rows(grid, header):
    """Body rows holding only an outcome label above rows of values, by row number."""
    rows = grid['rows']
    return {r: header_n(lone_label(rows[r - 1]))[0]
            for r in range(header['header_rows'] + 1, len(rows) + 1)
            if lone_label(rows[r - 1]) and any(map(_has_number, rows[r:]))}


def _body(grid, header):
    skipped = subheader_rows(grid, header)
    return [(r, row) for r, row in enumerate(grid['rows'], start=1)
            if r > header['header_rows'] and r not in skipped]


def read_layout(grid, header):
    """Which column holds arms, which holds outcomes, and whether rows are arms."""
    kinds = {c['col']: _column_kind(c) for c in header['columns']}
    named = {kind: next((col for col, k in kinds.items() if k == kind), None)
             for kind in ('arm', 'outcome')}
    candidate = named['arm'] or (1 if kinds and named['outcome'] != 1 else None)
    stubs = [row[candidate - 1] for _, row in _body(grid, header)] if candidate else []
    rows_are_arms = bool(candidate) and (
        named['arm'] is not None or any(match_rule(GROUP_RULES, header_n(s)[0]) for s in stubs))
    outcome_col = named['outcome'] or (1 if not rows_are_arms and kinds.get(1) == 'stub' else None)
    return {'kinds': kinds, 'arm_col': candidate if rows_are_arms else None,
            'outcome_col': outcome_col, 'rows_are_arms': rows_are_arms}


def _group_entries(grid, header, layout):
    count = header['header_rows']
    if layout['rows_are_arms']:
        col = layout['arm_col']
        rows = _body(grid, header)
        labeled = [header_n(row[col - 1]) for _, row in rows]
        filled = list(accumulate([label for label, _ in labeled], lambda left, text: text or left))
        found = [('row', r, label, n, r) for (r, _), label, (_, n) in zip(rows, filled, labeled)]
    else:
        found = [('col', c['col'], label_residue(' '.join(p for p in c['parts'][:-1] if p)
                                            if count > 1 else c['label']), c['n'], c['n_row'])
                 for c in header['columns']
                 if layout['kinds'][c['col']] is None
                 and not re.search(VETO, _squash(c['label']))]
    return [{'axis': axis, 'index': index, 'label': label, 'n': n, 'n_row': n_row,
             'group': match_rule(GROUP_RULES, label), 'source': 'rule', 'quote': label}
            for axis, index, label, n, n_row in found if label]


def find_groups(grid, header, overrides=None, respond=None, permission=None, endpoint=None,
                tracker=None, paper_id=None):
    """Arm labels in the rows or the header, each with its group role.

    Rows are arms when a column is headed Group, Condition, or Arm, or when a
    first-column label names a treatment or control group; blank arm cells
    take the arm above. Rows holding only an outcome label are not arms.
    Otherwise arms are read from the column headers
    without their stat and time words. Rules first; a model, when given, is
    asked only when the rules leave the table without both a treatment and a
    control, and may only choose from ``GROUP_ROLES``. ``overrides`` maps a
    label, or ``('col', n)`` / ``('row', n)``, to a role or None, and always
    wins. Each entry keeps its label as written, so arms stay apart.
    """
    entries = _group_entries(grid, header, read_layout(grid, header))
    keys = {k for e in entries for k in ((e['axis'], e['index']), e['label'])}
    human = _overrides(overrides, 'group', {'group': GROUP_ROLES}, keys)
    found = {e['group'] for e in entries if e['group']}
    asked = ([] if {'treatment', 'control'} <= found else
             list(dict.fromkeys(e['label'] for e in entries
                                if e['group'] is None and (e['axis'], e['index']) not in human
                                and e['label'] not in human)))
    model = {'respond': respond, 'permission': permission, 'endpoint': endpoint,
             'tracker': tracker, 'paper_id': paper_id}
    chosen, issue = _ask_model(asked, GROUP_ROLES, 'group label', model)
    return [_apply(_settle(e, 'group', asked, chosen, issue), human,
                   ((e['axis'], e['index']), e['label']))
            for e in entries]
