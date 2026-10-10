"""Single-case graph data: figure images, axis calibration, markers, phase lines, and the scdbayes CSV."""

import csv
import io
import re
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from .fulltext import X_TOLERANCE, _height, _row_kinds, find_gutter, word_rows
from .image_tables import table_image
from .tables import _follow, _joined, _row_text

SCD_COLUMNS = ('study', 'case', 'session', 'phase', 'outcome', 'trials', 'series', 'x_unit',
               'y_unit', 'outcome_type', 'design', 'intervention_goal', 'page', 'figure',
               'panel', 'calibration_id', 'confidence', 'checks')
DESIGNS = ('ab', 'reversal', 'multiple_baseline', 'multiple_probe', 'alternating',
           'changing_criterion')
OUTCOME_TYPES = ('continuous', 'count', 'binary', 'proportion', 'ordinal')
GOALS = ('increase', 'decrease')
CHOICES = {'design': DESIGNS, 'outcome_type': OUTCOME_TYPES, 'intervention_goal': GOALS}
FIGURE = re.compile(r'(?i)(?:figure|fig\.)\s*(\d+)'
                    r'(?=\s*(?:[.:|]|$)|\s*,\s*(?-i:[A-Z])|\s+(?-i:[A-Z]))')
PICTURES = ('image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/tiff')
PROSE_WORDS = 6
PROSE_GAP = 2
MIN_HEIGHT = 3
DARK = 128
AXIS_SHARE = 0.4
LINE_SHARE = 0.5
PANEL_GAP = 0.015
PANEL_SHARE = 0.12
CRITERION_SHARE = 0.15
HOLE_FILL = 0.8
EDGE_SHARE = 0.02
FLAGGED_CONFIDENCE = 0.5
# crops render at 200 dpi over 72-point PDF units
CROP_SCALE = 200 / 72


OUTCOME_CUES = (('proportion', re.compile(r'(?i)%|\bpercent\w*|\bproportion')),
                ('count', re.compile(r'(?i)\b(?:number|frequency|count|tally|rate)\b')),
                ('ordinal', re.compile(r'(?i)\b(?:rating|likert|rubric)\b')),
                ('binary', re.compile(r'(?i)\b(?:occurrence|yes/no|present/absent)\b')))


def suggest_outcome_type(caption='', words=()):
    """Outcome type of a single-case graph from its axis and caption words, and the evidence.

    Percent or proportion; number, frequency, count, tally, or rate; rating,
    Likert, or rubric; occurrence, yes/no, or present/absent, in that order. No cue gives
    'continuous' with no evidence, for a coder to check.
    """
    text = ' '.join([caption or '', *(w['text'] for w in words)])
    found = next(((kind, cue.search(text).group(0)) for kind, cue in OUTCOME_CUES
                  if cue.search(text)), None)
    return (found[0], [f"'{found[1]}' on the graph or caption"]) if found else ('continuous', [])


def _open_image(data):
    return Image.open(io.BytesIO(data))


def _png(picture):
    buffer = io.BytesIO()
    picture.save(buffer, format='PNG')
    return buffer.getvalue()


def _prose(row, height):
    """Six or more words without a gap over two line heights, so side-by-side boxes are not prose."""
    gaps = [b['x0'] - a['x1'] > PROSE_GAP * height for a, b in zip(row, row[1:])]
    return np.diff(np.flatnonzero([True] + gaps + [True])).max() >= PROSE_WORDS


def _figure_box(page, rows, i, height):
    prose = [k for k, row in enumerate(rows) if _prose(row, height)]
    end = _follow(rows, i, height, single=False)
    above = max([k for k in prose if k < i], default=None)
    below = min([k for k in prose if k >= end], default=None)
    ceiling = 0 if above is None else max(w['bottom'] for w in rows[above])
    stop = page.height if below is None else rows[below][0]['top']
    top = rows[i][0]['top']
    band = ((ceiling, top) if top - ceiling >= MIN_HEIGHT * height
            else (max(w['bottom'] for w in rows[end - 1]), stop))
    edges = [(w['x0'], w['x1']) for row in rows for w in row] + [
        (image['x0'], image['x1']) for image in page.images
        if image['top'] < band[1] and image['bottom'] > band[0]]
    box = (min(e[0] for e in edges), band[0], max(e[1] for e in edges), band[1])
    caption = _joined(_row_text(row) for row in rows[i:end])
    return (box, caption) if box[3] - box[1] >= MIN_HEIGHT * height else (None, caption)


def _column_views(page, words):
    """Rows of each column, with full-width rows in both, so a caption stays in its column."""
    rows = word_rows(words)
    gutter = find_gutter(page, words) if words else None
    if gutter is None:
        return [rows]
    kinds = _row_kinds(rows, gutter, _height(words))
    return [[row if kind == 'full' else part for kind, row in zip(kinds, rows)
             for part in [[w for w in row if side(w)]] if part]
            for side in (lambda w: w['x1'] <= gutter, lambda w: w['x0'] >= gutter)]


def figure_images(path):
    """Figures in a PDF or DOCX as ``figure``, ``pdf_page``, ``caption``, ``png`` bytes, and ``words``.

    A PDF figure is the page region above a 'Figure N' caption, up to the
    previous prose line, or below it when the space above is too short; a
    DOCX figure is an embedded picture, captioned by the 'Figure N'
    paragraphs in order. ``words`` are the PDF text-layer words inside the
    crop as ``x``, ``y``, ``x0``, ``x1``, ``top``, ``bottom`` (crop pixels), and
    ``text``; a DOCX picture has none.
    """
    if Path(path).suffix.lower() == '.docx':
        from .docx_reader import read_docx, read_docx_images
        captions = [block['text'] for block in read_docx(path)
                    if block['kind'] == 'paragraph' and FIGURE.match(block['text'])]
        pictures = [image for image in read_docx_images(path) if image['content_type'] in PICTURES]
        return [{'figure': number, 'pdf_page': None,
                 'caption': captions[number - 1] if number <= len(captions) else '',
                 'png': _png(_open_image(image['content']).convert('RGB')), 'words': []}
                for number, image in enumerate(pictures, start=1)]
    try:
        import pdfplumber
    except ImportError:
        raise ImportError('Figure crops require pdfplumber; reinstall pubextractor') from None
    with pdfplumber.open(str(path)) as document:
        pages = [(number, page, _column_views(page, words), words)
                 for number, page in enumerate(document.pages, start=1)
                 for words in [page.extract_words(x_tolerance=X_TOLERANCE)]]
        found = {}
        [found.setdefault((number, int(FIGURE.match(_row_text(rows[i])).group(1))),
                          (*_figure_box(page, rows, i, _height(words)), words))
         for number, page, views, words in pages for rows in views for i in range(len(rows))
         if FIGURE.match(_row_text(rows[i]))]
    return [{'figure': figure, 'pdf_page': number, 'caption': caption,
             'png': table_image(path, number, box),
             'words': [{'x': ((w['x0'] + w['x1']) / 2 - box[0]) * CROP_SCALE,
                        'y': ((w['top'] + w['bottom']) / 2 - box[1]) * CROP_SCALE,
                        'x0': (w['x0'] - box[0]) * CROP_SCALE, 'x1': (w['x1'] - box[0]) * CROP_SCALE,
                        'top': (w['top'] - box[1]) * CROP_SCALE,
                        'bottom': (w['bottom'] - box[1]) * CROP_SCALE, 'text': w['text']}
                       for w in words
                       if box[0] <= w['x0'] and w['x1'] <= box[2] and box[1] <= w['top']
                       and w['bottom'] <= box[3]]}
            for (number, figure), (box, caption, words) in found.items() if box]


def _fit(ticks):
    pixels, values = np.asarray(ticks, dtype=float).reshape(-1, 2).T
    if len(pixels) < 2 or np.ptp(pixels) == 0:
        raise ValueError('each axis needs two or more ticks at different pixels')
    scales = [values] + ([np.log10(values)] if len(pixels) > 2 and (values > 0).all() else [])
    fits = [np.polyfit(pixels, scale, 1) for scale in scales]
    misfits = [np.abs(np.polyval(fit, pixels) - scale).max() / (np.ptp(scale) or 1)
               for fit, scale in zip(fits, scales)]
    log = int(len(scales) > 1 and misfits[0] > 0.05 and misfits[1] < misfits[0])
    return {'slope': float(fits[log][0]), 'intercept': float(fits[log][1]), 'log': bool(log),
            'residual': float(misfits[log]),
            'range': (float(values.min()), float(values.max())),
            'centre': float(pixels.mean())}


def calibrate(x_ticks, y_ticks):
    """Linear pixel-to-value map per axis from two or more ``(pixel, value)`` ticks.

    An axis whose three or more ticks sit evenly on a log scale is flagged
    ``log`` and mapped on that scale. ``residual`` is the largest tick misfit
    as a share of the tick range; ``centre`` is the mean tick pixel.
    """
    return {'x': _fit(x_ticks), 'y': _fit(y_ticks),
            'id': f"{zlib.crc32(repr((list(x_ticks), list(y_ticks))).encode()):08x}"}


def _mask(png, color=None, tolerance=60):
    rgb = np.asarray(_open_image(png).convert('RGB'), dtype=float)
    target = (color if not isinstance(color, str)
              else [int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)])
    return (rgb.mean(axis=2) < DARK if color is None
            else np.linalg.norm(rgb - np.asarray(target, dtype=float), axis=2) < tolerance)


def _axes(dark):
    rows = np.flatnonzero(dark.mean(axis=1) >= AXIS_SHARE)
    cols = np.flatnonzero(dark.mean(axis=0) >= AXIS_SHARE)
    if not len(rows) or not len(cols):
        raise ValueError('no axis lines found')
    left, bottom = cols.min(), rows.max()
    thickness = int((rows > bottom - 6).sum())
    return (left, np.flatnonzero(dark[bottom]).max(), np.flatnonzero(dark[:, left]).min(), bottom,
            thickness)


def _box(mask, h, w):
    total = np.pad(mask.astype(int), ((h + 1, h), (w + 1, w))).cumsum(0).cumsum(1)
    return (total[2 * h + 1:, 2 * w + 1:] - total[:-2 * h - 1, 2 * w + 1:]
            - total[2 * h + 1:, :-2 * w - 1] + total[:-2 * h - 1, :-2 * w - 1])


def _parts(mask, radius=0):
    """Pixels, part index, sizes, centers, and half extents of the 4-connected parts of ``mask``.

    With ``radius``, each part also takes the pixels within that window of it.
    """
    ys, xs = np.nonzero(mask)
    empty = len(ys)
    index = np.full(np.add(mask.shape, 2 * radius + 2), empty)
    index[ys + radius + 1, xs + radius + 1] = np.arange(empty)
    near = [index[ys + radius + 1 + dy, xs + radius + 1 + dx]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    labels, spread = None, np.arange(empty + 1)
    while labels is None or (spread != labels).any():
        labels = spread
        spread = np.r_[np.minimum.reduce([labels[:-1], *[labels[n] for n in near]]), empty]
    ys, xs = np.nonzero(_box(mask, radius, radius) > 0)
    grown = np.minimum.reduce([labels[index[ys + 1 + dy, xs + 1 + dx]]
                               for dy in range(2 * radius + 1) for dx in range(2 * radius + 1)])
    _, part, counts = np.unique(grown, return_inverse=True, return_counts=True)
    cx, cy = np.bincount(part, xs) / counts, np.bincount(part, ys) / counts
    wide, tall = np.zeros(len(counts)), np.zeros(len(counts))
    np.maximum.at(wide, part, np.abs(xs - cx[part]))
    np.maximum.at(tall, part, np.abs(ys - cy[part]))
    return ys, xs, part, counts, cx, cy, wide, tall


def _blobs(png, mask, radius=None):
    """Centers, box fill share, and mean color of markers left by an opening of ``mask``.

    Holes enclosed by ink are filled first where, joined across thin lines,
    they are compact, so open markers count as filled. A blob covering k
    times the median marker area becomes k points along its major axis.
    """
    left, right, top, bottom, thickness = _axes(_mask(png))
    radius = radius or max(2, 2 * thickness)
    reach = 4 * radius
    outside = ~mask & (_box(mask, reach, reach) == 0)
    outside[[0, -1]] |= ~mask[[0, -1]]
    outside[:, [0, -1]] |= ~mask[:, [0, -1]]
    grown = None
    while grown is None or (grown != outside).any():
        outside = outside if grown is None else grown
        edged = np.pad(outside, 1)
        grown = ~mask & (edged[1:-1, 1:-1] | edged[:-2, 1:-1] | edged[2:, 1:-1]
                         | edged[1:-1, :-2] | edged[1:-1, 2:])
    holes = ~mask & ~outside
    ys, xs, hole, counts, _, _, wide, tall = _parts(_box(holes, radius, radius) > 0)
    compact = ((counts >= HOLE_FILL * (2 * wide + 1) * (2 * tall + 1))
               & (wide + 1 <= 2 * (tall + 1)) & (tall + 1 <= 2 * (wide + 1)))
    joined = np.zeros(mask.shape, dtype=bool)
    joined[ys[compact[hole]], xs[compact[hole]]] = True
    mask = mask | holes & joined
    core = _box(mask, radius, radius) == (2 * radius + 1) ** 2
    ys, xs, blob, counts, cx, cy, wide, tall = _parts(core, radius)
    dx, dy = xs - cx[blob], ys - cy[blob]
    angle = 0.5 * np.arctan2(2 * np.bincount(blob, dx * dy),
                             np.bincount(blob, dx * dx) - np.bincount(blob, dy * dy))
    along = dx * np.cos(angle)[blob] + dy * np.sin(angle)[blob]
    low, high = np.full(len(counts), np.inf), np.full(len(counts), -np.inf)
    np.minimum.at(low, blob, along)
    np.maximum.at(high, blob, along)
    rgb = np.asarray(_open_image(png).convert('RGB'), dtype=float)[ys, xs]
    colors = np.stack([np.bincount(blob, rgb[:, c]) / counts for c in range(3)], axis=1)
    inside = (cx > left + radius) & (cx <= right + radius) & (cy >= top - radius) & (
        cy <= bottom + radius)
    single = np.median(counts[inside]) if inside.any() else np.inf
    half = np.median((high - low)[inside]) / 2 if inside.any() else 0
    k = np.maximum(np.round(counts / single), 1).astype(int)
    keep = inside & (wide + 1 <= 2 * k * (tall + 1)) & (tall + 1 <= 2 * k * (wide + 1))
    each = np.repeat(np.flatnonzero(keep), k[keep])
    step = np.arange(len(each)) - np.repeat(np.cumsum(k[keep]) - k[keep], k[keep])
    shift = np.where(k[each] > 1, low[each] + half + step * (high - low - 2 * half)[each]
                     / np.maximum(k[each] - 1, 1), 0)
    px, py = cx[each] + shift * np.cos(angle)[each], cy[each] + shift * np.sin(angle)[each]
    order = np.argsort(px, kind='stable')
    fill = counts / ((2 * wide + 1) * (2 * tall + 1))
    return px[order], py[order], fill[each][order], colors[each][order]


def find_points(png, colour=None, radius=None):
    """Pixel centers ``(x, y)`` of filled or open markers inside the axes, ordered by x.

    Markers are dark pixels, or pixels within reach of ``colour`` (hex or
    RGB). An opening by ``radius`` (twice the x-axis thickness, at least 2)
    removes lines thinner than the markers, and each remaining blob of
    roughly square extent is one point; touching markers split by area.
    """
    cx, cy, _, _ = _blobs(png, _mask(png, colour), radius)
    return [(float(x), float(y)) for x, y in zip(cx, cy)]


def find_series(png, radius=None):
    """Markers as ``(x, y, series)``, the series named by shape, color, or both where they vary.

    Shape comes from how much of its box a marker fills (square, circle,
    triangle); color is the marker's mean color on a coarse grid.
    """
    cx, cy, fill, colors = _blobs(png, ~_mask(png, (255, 255, 255), 120), radius)
    shapes = np.select([fill >= 0.9, fill >= 0.65], ['square', 'circle'], 'triangle')
    hexes = ['#' + ''.join(f"{int(v) // 64 * 64 + 32:02x}" for v in color) for color in colors]
    by_shape, by_color = len(set(shapes)) > 1, len(set(hexes)) > 1
    return [(float(x), float(y), ' '.join([shape] * (by_shape or not by_color) + [hue] * by_color))
            for x, y, shape, hue in zip(cx, cy, shapes, hexes)]


def find_phase_lines(png):
    """Pixel x of vertical lines, solid or dashed, inked over half the axes height or more."""
    dark = _mask(png)
    left, right, top, bottom, _ = _axes(dark)
    fill = dark[top:bottom].mean(axis=0)
    cols = np.array([c for c in np.flatnonzero(fill >= LINE_SHARE) if left + 3 < c < right - 3])
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1) if len(cols) else []
    return [float(group.mean()) for group in groups]


def find_criterion_lines(png):
    """Horizontal segments inside the axes, at least 15% of their width, as ``(y, x0, x1)``.

    Gaps up to six times the axis thickness along a row are bridged, so
    dashed lines count as solid.
    """
    dark = _mask(png)
    left, right, top, bottom, thickness = _axes(dark)
    reach = 3 * thickness
    inner = (_box(_box(dark[top + 3:bottom - 3, left + 3:right], 0, reach) > 0, 0, reach)
             == 2 * reach + 1).astype(int)
    edges = [np.flatnonzero(np.diff(np.r_[0, row, 0])) for row in inner]
    runs = [(top + 3 + r, left + 3 + e[i], left + 2 + e[i + 1]) for r, e in enumerate(edges)
            for i in range(0, len(e), 2) if e[i + 1] - e[i] >= CRITERION_SHARE * (right - left)]
    spans = {}
    [spans.setdefault((x0 // 4, x1 // 4), []).append((y, x0, x1)) for y, x0, x1 in runs]
    return sorted((float(np.mean([r[0] for r in part])), float(min(r[1] for r in part)),
                   float(max(r[2] for r in part)))
                  for group in spans.values()
                  for part in np.split(np.array(group), np.flatnonzero(
                      np.diff([r[0] for r in group]) > 2) + 1))


def split_panels(png, count=None):
    """Stacked panels split at blank rows, or into ``count`` equal bands.

    Each panel has ``panel``, ``top`` and ``bottom`` (pixel rows of the
    crop), and ``png``.
    """
    dark = _mask(png)
    height = len(dark)
    rows = np.flatnonzero(dark.any(axis=1))
    if not len(rows):
        raise ValueError('no axis lines found')
    breaks = np.flatnonzero(np.diff(rows) > max(PANEL_GAP * height, 4))
    found = [(int(a), int(b)) for a, b in zip(rows[np.r_[0, breaks + 1]], rows[np.r_[breaks, -1]])
             if b - a >= PANEL_SHARE * height]
    bands = ([(round(i * height / count), round((i + 1) * height / count) - 1)
              for i in range(count)] if count else found)
    picture = _open_image(png)
    return [{'panel': number, 'top': top, 'bottom': bottom,
             'png': _png(picture.crop((0, top, picture.width, bottom)))}
            for number, (a, b) in enumerate(bands, start=1)
            for top, bottom in [(max(a - 2, 0), min(b + 3, height))]]


def read_graph(png, design='ab', count=None):
    """Panels of a figure as the design reads them, in pixels of the whole figure.

    Multiple-baseline and multiple-probe figures split into stacked panels
    (``count`` equal bands when given); other designs keep one panel. Each
    panel has ``panel``, ``top``, ``bottom``, ``axis`` (its x-axis row),
    ``area`` (left, right, top, bottom of its axes), ``points``, phase
    ``lines``, and ``criteria`` (changing criterion only).
    Alternating points carry their series as a third item and have no phase
    lines.
    """
    stacked = design in ('multiple_baseline', 'multiple_probe')
    whole = {'panel': 1, 'top': 0, 'bottom': _open_image(png).height, 'png': png}
    panels = (split_panels(png, count) if stacked else []) or [whole]
    return [{'panel': p['panel'], 'top': p['top'], 'bottom': p['bottom'],
             'axis': float(box[3] + p['top']),
             'area': (float(box[0]), float(box[1]), float(box[2] + p['top']),
                      float(box[3] + p['top'])),
             'points': [(x, y + p['top'], *rest) for x, y, *rest in
                        (find_series(p['png']) if design == 'alternating'
                         else find_points(p['png']))],
             'lines': [] if design == 'alternating' else find_phase_lines(p['png']),
             'criteria': ([(y + p['top'], x0, x1) for y, x0, x1 in find_criterion_lines(p['png'])]
                          if design == 'changing_criterion' else [])}
            for p in panels for box in [_axes(_mask(p['png']))]]


def _beyond(values, low, high):
    return np.abs(values - np.clip(values, low, high)) > EDGE_SHARE * (high - low)


def _values(axis, pixels):
    values = axis['slope'] * np.asarray(pixels, dtype=float) + axis['intercept']
    return 10 ** values if axis['log'] else values


def graph_rows(points, calibration, phase_lines=(), outcome_type='continuous', levels=None,
               criterion_lines=None, area=None, **fields):
    """scdbayes rows from marker pixels: values, sessions, phases, and snapped outcomes.

    Points are ordered by x; ``session`` is the x value rounded. Phases run
    A1, B1, A2, ... across ``phase_lines`` (pixel x); a point given as
    ``(x, y, series)`` takes its series as both ``series`` and ``phase``.
    Outcomes snap by ``outcome_type``: count to whole numbers, binary to 0/1,
    proportion to 0-1, ordinal to the nearest of ``levels`` (whole numbers
    when not given). A proportion axis topping 1.5 is read as percent and
    divided by 100 when its top is 50 or more or ``y_unit`` has '%';
    otherwise the values stay and ``checks`` gets ``check_scale``. With
    ``criterion_lines`` (``(y, x0, x1)`` pixels) each row gains a
    ``criterion`` value from the line over its x. Other columns of
    ``SCD_COLUMNS`` come from ``fields``; ``confidence`` is one minus the
    larger calibration residual. A point more than 2% beyond ``area``
    (left, right, top, bottom pixels), or beyond the tick values when
    ``area`` is None, or before session 1, is kept with ``outside_axes``
    in ``checks`` and confidence 0.5. With no phase lines every row is A1.
    """
    base = {**dict.fromkeys(SCD_COLUMNS, ''), 'series': 1, 'panel': 1, 'x_unit': 'session',
            'outcome_type': outcome_type, 'calibration_id': calibration['id'],
            'confidence': round(1 - max(calibration['x']['residual'],
                                        calibration['y']['residual']), 3), **fields}
    problems = ([f"unknown columns {sorted(set(fields) - set(SCD_COLUMNS))}"]
                if set(fields) - set(SCD_COLUMNS) else []) + [
        f"{key} {base[key]!r} is not one of {list(choices)}"
        for key, choices in CHOICES.items() if base[key] and base[key] not in choices]
    if problems:
        raise ValueError('; '.join(problems))
    marks = sorted(points, key=lambda point: point[0])
    x, y = np.asarray([mark[:2] for mark in marks], dtype=float).reshape(-1, 2).T
    series = [mark[2] if len(mark) > 2 else None for mark in marks]
    sessions, raw = _values(calibration['x'], x), _values(calibration['y'], y)
    phases = np.searchsorted(np.sort(np.asarray(phase_lines, dtype=float)), x)
    steps = np.asarray(levels if levels is not None
                       else np.arange(np.floor(raw.min(initial=0)), np.ceil(raw.max(initial=0)) + 1),
                       dtype=float)
    top = calibration['y']['range'][1]
    percent = top > 1.5 and (top >= 50 or '%' in str(base['y_unit']))
    scale = outcome_type == 'proportion' and top > 1.5 and not percent
    outside = ((_beyond(x, *area[:2]) | _beyond(y, *area[2:])) if area else
               _beyond(sessions, *calibration['x']['range'])
               | _beyond(raw, *calibration['y']['range'])) | (sessions.round() < 1)
    snapped = {'continuous': raw.round(4), 'count': np.maximum(raw.round(), 0),
               'binary': (raw >= 0.5).astype(float),
               'proportion': (raw if scale else np.clip(raw / (100 if percent else 1), 0, 1))
               .round(4),
               'ordinal': steps[np.abs(raw[:, None] - steps).argmin(axis=1)] if len(raw)
               else raw}[outcome_type]
    criteria = [{'criterion': next((round(float(_values(calibration['y'], [level])[0]), 4)
                                    for level, x0, x1 in criterion_lines if x0 <= at <= x1), '')}
                if criterion_lines is not None else {} for at in x]
    whole = outcome_type in ('count', 'binary')
    return [{**base, 'session': int(round(s)),
             'phase': label or f"{'AB'[k % 2]}{k // 2 + 1}", 'series': label or base['series'],
             'outcome': int(v) if whole else float(v),
             'confidence': FLAGGED_CONFIDENCE if out else base['confidence'],
             'checks': '; '.join(['outside_axes'] * out + ['check_scale'] * scale), **criterion}
            for s, k, v, label, criterion, out in zip(sessions, phases, snapped, series, criteria,
                                                      outside.tolist())]


def design_rows(panels, calibration, design='ab', cases=(), **fields):
    """Rows for every panel of ``read_graph``, one case per panel.

    The y map, calibrated on the panel holding the y ticks, shifts to each
    other panel by the offset between their x-axis rows. ``cases`` names
    the panels in order; unnamed panels are ``case<N>``.
    """
    center = calibration['y']['centre']
    reference = next((p for p in panels if p['top'] <= center <= p['bottom']), panels[0])
    return [row for p in panels for shift in [p['axis'] - reference['axis']]
            for row in graph_rows(
                [(x, y - shift, *rest) for x, y, *rest in p['points']], calibration, p['lines'],
                area=(*p['area'][:2], p['area'][2] - shift, p['area'][3] - shift)
                if 'area' in p else None,
                criterion_lines=([(y - shift, x0, x1) for y, x0, x1 in p['criteria']]
                                 if design == 'changing_criterion' else None),
                design=design, panel=p['panel'],
                case=cases[p['panel'] - 1] if p['panel'] <= len(cases) else f"case{p['panel']}",
                **fields)]


def to_scdbayes(rows, path):
    """Write graph rows to a CSV whose case, session, and phase columns scdbayes detects."""
    extra = dict.fromkeys(key for row in rows for key in row if key not in SCD_COLUMNS)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*SCD_COLUMNS, *extra])
        writer.writeheader()
        writer.writerows(rows)
    return path
