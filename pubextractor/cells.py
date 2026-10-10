import re

from .statistics import MINUS_SIGNS

NUM = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+'

MISSING = re.compile(r'[-‒–—―−]{1,3}|n/?a|n\.?r\.?|not reported',
                     re.IGNORECASE)

# a single trailing letter counts as a footnote marker only when no letter follows
MARKER = re.compile(r'(?<=[\d)%])\s?(?:\*{1,3}|[†‡§]+|[a-z](?![a-z]))')

CELL_PATTERNS = (
    ('n_percent', rf'(?P<value>{NUM})\s*\(\s*(?P<second>{NUM})\s*%\s*\)'),
    ('mean_sd', rf'(?P<value>{NUM})\s*\(\s*(?P<second>{NUM})\s*\)'),
    ('mean_sd', rf'(?P<value>{NUM})\s*(?:±|\+/-)\s*(?P<second>{NUM})'),
    ('percent', rf'(?P<value>{NUM})\s*%'),
    ('number', rf'(?P<value>{NUM})'),
)

N_LABEL = re.compile(r'[,;]?\s*\(?\s*\b[Nn]\s*=\s*(\d{1,3}(?:,\d{3})+|\d+)\s*\)?')


def _number(token):
    return float(token.replace(',', ''))


def _cell(kind, value=None, second=None, markers=()):
    return {'kind': kind, 'value': value, 'second': second, 'markers': list(markers)}


def parse_cell(text):
    raw = ' '.join((text or '').split())
    if not raw:
        return _cell('empty')
    if MISSING.fullmatch(raw):
        return _cell('missing')
    clean = raw.translate(MINUS_SIGNS)
    bare = MARKER.sub('', clean).strip()
    found = next(((kind, match) for kind, pattern in CELL_PATTERNS
                  for match in [re.fullmatch(pattern, bare)] if match), None)
    if found is None:
        return _cell('text')
    kind, match = found
    second = match.groupdict().get('second')
    return _cell(kind, _number(match['value']), _number(second) if second else None,
                 [marker.strip() for marker in MARKER.findall(clean)])


def header_n(text):
    raw = ' '.join((text or '').split())
    match = N_LABEL.search(raw)
    if match is None:
        return raw, None
    label = ' '.join((raw[:match.start()] + ' ' + raw[match.end():]).split()).strip(' ,;')
    return label, int(match.group(1).replace(',', ''))
