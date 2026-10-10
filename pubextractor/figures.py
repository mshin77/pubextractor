"""Bar charts, forest plots, scatter plots, and flow diagrams read into records."""

import re

import numpy as np

from .graphs import _axes, _box, _mask, _open_image, _parts, _values, find_phase_lines, find_points
from .output import COLUMNS
from .statistics import find_statistics
from .table_roles import GROUP_RULES, match_rule
from .tables import _slug

FIGURE_TYPES = ('scd_line', 'bar', 'scatter', 'forest', 'flow', 'other')
ERROR_BARS = ('se', 'sd', 'ci')
NOT_LINE = {'correlational': 'scatter'}
CAPTION_TYPES = (
    ('flow', r'\b(?:flow|consort|enrol\w*|allocation)\b'),
    ('forest', r'\bforest\b'),
    ('scatter', r'\b(?:scatter\w*|correlations?|relationship)\b'),
    ('scd_line', r'\b(?:sessions?|baseline|probes?|phases?|single[- ]case)\b'),
    ('bar', r'\b(?:means?|bars?)\b'),
)
WHITE = (255, 255, 255)
PAPER = 40
BAR_SHARE = 0.03
WHISKER_SPAN = 3
LOOSE_INK = 0.2
UNLIKE = 60
Z95 = 1.96
LABEL_WORDS = 5
DENSE = 0.15
MIN_TALL = 0.05
LINE_GAP = 0.8
WORD_GAP = 1.5
TICK_REACH = 60
TICK_LONG = 20
TICK_CLEAR = 6
TICK_GAP = 20
TICK_TOL = 3
TICKS = 4
AXIS_RUN = 0.08
TYPED_N = re.compile(r'\(?\bn\s*=\s*(\d+)\)?', re.I)


def _solid(png, mask):
    """Axes box and the blobs of ``mask`` thicker than the axis lines, with which lie inside."""
    left, right, top, bottom, thickness = _axes(_mask(png))
    radius = max(2, 2 * thickness)
    ys, xs, part, _, cx, cy, wide, tall = _parts(_box(mask, radius, radius)
                                                 == (2 * radius + 1) ** 2, radius)
    inside = (cx > left + radius) & (cx < right) & (cy > top) & (cy < bottom)
    return (left, right, top, bottom), ys, xs, part, cx, cy, wide, inside


def _bars(png):
    """Center, top row, and color of each bar, left to right.

    A bar is a run of columns inside the axes at least 3% of the x axis wide
    whose ink covers 15% of each column's height and of its lower half, so
    solid and hatched fills both count.
    """
    rgb = np.asarray(_open_image(png).convert('RGB'), dtype=float)
    ink = ~_mask(png, WHITE, PAPER)
    left, right, top, bottom, thickness = _axes(_mask(png))
    start, floor = left + thickness + 1, bottom - thickness + 1
    inner = ink[top:floor, start:right]
    height = len(inner)
    head = np.where(inner.any(axis=0), inner.argmax(axis=0), height)
    low = np.arange(height)[:, None] >= (head + height) / 2
    dense = ((inner.sum(axis=0) >= DENSE * (height - head))
             & ((inner & low).sum(axis=0) >= DENSE * low.sum(axis=0))
             & (height - head >= MIN_TALL * height))
    runs = np.flatnonzero(np.diff(np.r_[0, dense.astype(int), 0])).reshape(-1, 2)
    runs = runs[runs[:, 1] - runs[:, 0] >= BAR_SHARE * (right - left)]
    heads = [top + int(np.median(head[a:b])) for a, b in runs]
    return [(start + (a + b - 1) / 2, h,
             np.median(rgb[h:floor, start + a:start + b][ink[h:floor, start + a:start + b]], axis=0))
            for (a, b), h in zip(runs, heads)]


def _spans(dark, cx, cy):
    """First and last pixel of the ink run along each blob's row, through its center."""
    rows = [dark[max(round(y) - 1, 0):round(y) + 2].any(axis=0) for y in cy]
    gaps = [np.flatnonzero(~row) for row in rows]
    return (np.array([g[g < x].max(initial=-1) + 1 for g, x in zip(gaps, cx)]),
            np.array([g[g > x].min(initial=len(row)) - 1 for g, x, row in zip(gaps, cx, rows)]))


def _record(place, stat, value, quote, **keys):
    return {**dict.fromkeys(COLUMNS), 'paper_id': place[0], 'value': value, 'layer': 'cv',
            'status': 'review', 'quote': quote, 'pdf_page': place[2],
            'table': 'figure' if place[1] is None else f'figure {place[1]}', 'section': '',
            'stat': stat, 'candidates': 0, **keys}


def figure_type(png, caption='', design=None):
    """'scd_line', 'bar', 'scatter', 'forest', 'flow', or 'other' as a suggestion for the coder.

    Caption words decide first; otherwise a confirmed single-case design means
    a line graph, and for other designs no axes means a flow diagram, solid
    or hatched blocks wider than 3% of the x axis mean bars, a vertical line with
    markers on horizontal whiskers three times their width means a forest
    plot, and markers with little other ink inside the axes mean a scatter.
    A paper of another confirmed design never gets a line graph from the image alone.
    A line graph also needs both axes with four evenly spaced ticks, with labels
    on the y axis, and connected markers or phase lines; otherwise the
    suggestion is 'other' (kept with its caption).
    """
    worded = [kind for kind, pattern in CAPTION_TYPES if re.search(pattern, caption or '', re.I)]
    drawn = worded[0] if worded else 'scd_line' if design == 'single_case' else _drawn_type(png)
    kind = (NOT_LINE.get(design, 'bar') if drawn == 'scd_line' and not worded
            and design not in (None, 'single_case') else drawn)
    dark = _mask(png) if kind == 'scd_line' else None
    lines = dark is None or _ticked_axes(dark) and bool(
        find_phase_lines(png) or len(find_points(png)) >= 3 and _joined(png, dark))
    return kind if lines else 'other'


def _runs(dark):
    """Length of the horizontal ink run through each pixel."""
    flat = np.pad(dark, ((0, 0), (0, 1))).ravel()
    edges = np.diff(np.r_[0, flat.astype(int), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    found = np.zeros(flat.size, dtype=int)
    found[flat] = np.repeat(lengths, lengths)
    return found.reshape(dark.shape[0], -1)[:, :-1]


def _even(ticks):
    """Most ticks in one run spaced evenly at least 20 pixels apart."""
    ticks = np.asarray(ticks, dtype=float)
    a, b = np.triu_indices(len(ticks), 1)
    keep = ticks[b] - ticks[a] >= TICK_GAP
    if not keep.any():
        return min(len(ticks), 1)
    targets = ticks[a[keep], None] + (ticks[b] - ticks[a])[keep, None] * np.arange(1, len(ticks) + 1)
    hits = np.abs(targets[..., None] - ticks).min(axis=-1) <= TICK_TOL
    return int(hits.argmin(axis=1).max()) + 1


def _stubs(stub, dark, c, with_labels):
    idx = np.flatnonzero(stub[:, c])
    centres = [round(float(g.mean())) for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
               if len(g) <= 4]
    return tuple(y for y in centres if not with_labels or dark[max(y - 6, 0):y + 7,
                                                            max(c - TICK_REACH, 0):max(c - TICK_LONG, 0)].any())


def _ticks(dark, with_labels):
    """Most evenly spaced tick marks on one vertical line, counting only ticks with labels if asked.

    A tick is a short stub across a line at least 8% of the image tall, with
    no other ink within 6 pixels along the line; a tick with a label has ink 20 to
    60 pixels to its left.
    """
    across, along = _runs(dark), _runs(dark.T).T
    line = along >= AXIS_RUN * dark.shape[0]
    rows = _box(dark, 4, TICK_CLEAR) - _box(dark, 1, TICK_CLEAR)
    core = _box(dark, 4, 2) - _box(dark, 1, 2)
    stub = line & (across >= 4) & (across <= TICK_LONG) & (rows == core)
    found = {_stubs(stub, dark, c, with_labels)
             for c in np.flatnonzero(stub.sum(axis=0) >= TICKS)}
    return max((_even(ticks) for ticks in found), default=0)


def _ticked_axes(dark):
    """True when a vertical and a horizontal line each carry four evenly spaced ticks, the
    vertical one with labels."""
    return min(_ticks(dark, True), _ticks(dark.T, False)) >= TICKS


def _joined(png, dark):
    """True when line ink inside the axes links the markers rather than leaving them loose."""
    (left, right, top, bottom), ys, xs, part, cx, cy, wide, inside = _solid(png, dark)
    near = np.zeros(dark.shape, dtype=bool)
    near[ys[inside[part]], xs[inside[part]]] = True
    loose = (dark & ~(_box(near, 2, 2) > 0))[top + 4:bottom - 4, left + 4:right - 4]
    return loose.sum() >= LOOSE_INK * near.sum()


def _drawn_type(png):
    try:
        bars = _bars(png)
    except ValueError:
        return 'flow'
    if bars:
        return 'bar'
    dark = _mask(png)
    (left, right, top, bottom), ys, xs, part, cx, cy, wide, inside = _solid(png, dark)
    x0, x1 = _spans(dark, cx[inside], cy[inside])
    if inside.any() and find_phase_lines(png) and np.median((x1 - x0 + 1) / (2 * wide[inside] + 1)) >= WHISKER_SPAN:
        return 'forest'
    return 'scd_line' if _joined(png, dark) else 'scatter'


def read_bars(png, calibration, names=(), error='se', paper_id=None, figure=None, pdf_page=None):
    """Mean and error records per bar, left to right, from a calibrated bar chart.

    Bars are solid or hatched blocks inside the axes wider than 3% of the x
    axis. The whisker runs up from the bar top to its last ink pixel and down
    through pixels unlike the bar color; a whisker the color of the bar
    mirrors its upper end. ``names`` are 'arm' or 'arm / outcome / time' in
    bar order, each optionally with 'n = N', which becomes an n record with
    layer 'human'; ``error`` is 'se', 'sd', or 'ci' (95%), and a CI gives se =
    half-width / 1.96. Records have layer 'cv', status 'review', and table
    'figure N', so ``build_comparisons`` pairs them as table values; an SE
    becomes an SD at effect sizes from the arm n. A bar with no whisker gives
    an error record with value None and 'no_error_bars' in its quote, so
    effect sizes report ``sd_missing``.
    """
    if error not in ERROR_BARS:
        raise ValueError(f"error must be one of {list(ERROR_BARS)}, not {error!r}")
    rgb = np.asarray(_open_image(png).convert('RGB'), dtype=float)
    ink = ~_mask(png, WHITE, PAPER)
    place = (paper_id, figure, pdf_page)
    typed = [TYPED_N.search(name) for name in names]
    labels = [[piece.strip() or None for piece in TYPED_N.sub('', name).split('/')] + [None, None]
              for name in names]
    found = []
    for k, (center, head, color) in enumerate(_bars(png)):
        c = round(center)
        above = np.flatnonzero(~ink[:head, c - 1:c + 2].any(axis=1))
        up = above.max() + 1 if len(above) else 0
        unlike = np.linalg.norm(rgb[head:, c - 1:c + 2] - color, axis=2).max(axis=1) > UNLIKE
        down = head + int(np.argmin(unlike)) - 1
        low = down if down > head + 1 else 2 * head - up
        mean, top_end, low_end = _values(calibration['y'], [head, up, low])
        bare = up >= head
        half = None if bare else (top_end - low_end) / 2 / (Z95 if error == 'ci' else 1)
        arm, outcome, time = labels[k][:3] if k < len(labels) else (f'bar {k + 1}', None, None)
        quote = (f"Figure {figure or ''} bar {arm}: mean {mean:.2f}, "
                 + ('no_error_bars' if bare else f"{error.upper()} bar {low_end:.2f} to "
                    f"{top_end:.2f}" + {'ci': '; se = half-width / 1.96',
                                        'se': '; sd = se x sqrt(n) at effect sizes',
                                        'sd': ''}[error]))
        keys = {'arm': arm, 'outcome': outcome, 'time': time,
                'group': match_rule(GROUP_RULES, arm)}
        found += [_record(place, stat, None if value is None else round(float(value), 4), quote,
                          field='_'.join(_slug(p) for p in (outcome, arm, time, stat) if p), **keys)
                  for stat, value in (('mean', mean), ('sd' if error == 'sd' else 'se', half))]
        n = typed[k] if k < len(typed) else None
        found += [_record(place, 'n', int(n.group(1)), f"Bar {arm}: {n.group(0)} typed by the coder",
                          layer='human', status='extracted',
                          field='_'.join(_slug(p) for p in (outcome, arm, time, 'n') if p), **keys)
                  ] if n else []
    return found


def read_forest(png, calibration, labels=(), words=(), paper_id=None, figure=None, pdf_page=None):
    """Effect, lower, and upper records per forest-plot row, top to bottom.

    The effect is the marker center on the calibrated x axis and the limits
    are the ends of the ink run along its row. Rows are named by ``labels``,
    else by ``words`` (``x``, ``y``, ``text`` in figure pixels) left of the
    axes within half a row, else 'row N'. Records have layer 'soma'.
    """
    dark = _mask(png)
    (left, *_), _, _, _, cx, cy, _, inside = _solid(png, dark)
    order = np.flatnonzero(inside)[np.argsort(cy[inside])]
    x0, x1 = _spans(dark, cx[order], cy[order])
    reach = np.median(np.diff(cy[order])) / 2 if len(order) > 1 else np.inf
    place = (paper_id, figure, pdf_page)
    found = []
    for k, (b, low, high) in enumerate(zip(order, x0, x1)):
        named = ' '.join(w['text'] for w in words if w['x'] < left and abs(w['y'] - cy[b]) < reach)
        study = labels[k] if k < len(labels) else named or f'row {k + 1}'
        values = _values(calibration['x'], [cx[b], low, high])
        quote = f"Figure {figure or ''} row {study}: {values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}]"
        found += [_record(place, stat, round(float(value), 4), quote, study=study, layer='soma',
                          field=f'{_slug(study)}_{stat}')
                  for stat, value in zip(('effect', 'lower', 'upper'), values)]
    return found


def read_scatter(png, calibration, names=('x', 'y'), paper_id=None, figure=None, pdf_page=None):
    """``points`` as calibrated (x, y) pairs and ``records`` of n and Pearson r (from three points)."""
    x, y = np.asarray(find_points(png), dtype=float).reshape(-1, 2).T
    xs, ys = _values(calibration['x'], x), _values(calibration['y'], y)
    r = round(float(np.corrcoef(xs, ys)[0, 1]), 4) if len(xs) > 2 else None
    pair = '_'.join(_slug(name) for name in names)
    quote = f"Figure {figure or ''}: {len(xs)} points, r = {r}"
    place = (paper_id, figure, pdf_page)
    return {'points': [(round(float(a), 4), round(float(b), 4)) for a, b in zip(xs, ys)],
            'records': [_record(place, stat, value, quote, field=f'{pair}_{stat}',
                                outcome=' vs '.join(names))
                        for stat, value in (('n', len(xs)), ('r', r))]}


def _flow_boxes(words):
    """Text of each box: words joined across line gaps under 0.8 and word gaps under 1.5 line heights."""
    x0, x1, top, bottom = (np.array([w[key] for w in words], dtype=float)
                           for key in ('x0', 'x1', 'top', 'bottom'))
    height = np.median(bottom - top)
    across = np.maximum(x0[None] - x1[:, None], x0[:, None] - x1[None])
    down = np.maximum(top[None] - bottom[:, None], top[:, None] - bottom[None])
    reach = ((across <= WORD_GAP * height) & (down <= LINE_GAP * height)).astype(int)
    for _ in range(int(np.log2(len(words))) + 1):
        reach = np.minimum(reach @ reach, 1)
    box = reach.argmax(axis=0)
    order = sorted(set(box), key=lambda b: (top[box == b].min(), x0[box == b].min()))
    return [' '.join(words[i]['text'] for i in sorted(np.flatnonzero(box == b),
                                                       key=lambda i: (round(top[i] / height), x0[i])))
            for b in order]


def read_flow(text='', paper_id=None, figure=None, pdf_page=None, words=()):
    """n records from 'n = N' in a flow diagram, each named by up to five words before it.

    ``words`` with ``x0``, ``x1``, ``top``, and ``bottom`` (as ``figure_images``
    gives) are grouped into boxes by their gaps, so boxes side by side keep
    their own words; otherwise ``text`` is read as one box.
    """
    return [r for box in (_flow_boxes(words) if len(words) else [text])
            for r in _flow_box(box, paper_id, figure, pdf_page)]


def _flow_box(text, paper_id, figure, pdf_page):
    found = [s for s in find_statistics(text) if s['stat'] == 'n']
    starts = [0] + [s['end'] for s in found[:-1]]
    labels = [' '.join(re.findall(r'[A-Za-z][\w-]*', text[a:s['start']])[-LABEL_WORDS:])
              for a, s in zip(starts, found)]
    place = (paper_id, figure, pdf_page)
    return [_record(place, 'n', int(s['value']), f"{label} ({s['text']})", layer='regex',
                    field='_'.join(filter(None, (_slug(label), 'n'))),
                    group=match_rule(GROUP_RULES, label))
            for label, s in zip(labels, found)]
