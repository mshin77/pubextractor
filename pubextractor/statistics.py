import re

# a comma is thousands before three digits and a decimal before one or two, so 'n = 25, M' stops at 25
COUNT = r'(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+)'
NUMBER = r'(-?(?:\d{1,3}(?:,\d{3})+|\d+,\d{1,2}(?!\d)|\d+)(?:\.\d+)?|-?\.\d+)'

PATTERNS = {
    'n': rf'(?<![\w.])[Nn]\s*=\s*{COUNT}',
    'mean': rf'(?<![\w.])(?:M|Mean)\s*=\s*{NUMBER}',
    'sd': rf'(?<![\w.])SD\s*=\s*{NUMBER}',
    'se': rf'(?<![\w.])SE\s*=\s*{NUMBER}',
    'mse': rf'(?<![\w.])MSE\s*=\s*{NUMBER}',
    't': rf'(?<![\w.])t\s*\(\s*(\d+(?:\.\d+)?)\s*\)\s*=\s*{NUMBER}',
    'f': rf'(?<![\w.])F\s*\(\s*(\d+)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*=\s*{NUMBER}',
    'r': rf'(?<![\w.])r\s*(?:\(\s*(\d+)\s*\))?\s*=\s*{NUMBER}',
    'p': rf'(?<![\w.])p\s*([<>=\u2264\u2265])\s*{NUMBER}',
}

WITH_DF = {'t', 'f', 'r'}

# U+2212 and U+2013 are single characters, so positions in the cleaned text match the original
MINUS_SIGNS = str.maketrans({'\u2212': '-', '\u2013': '-'})


def _number(token):
    return float(re.sub(r',(\d{1,2})$', r'.\1', token).replace(',', ''))


def _found(stat, match, text):
    groups = match.groups()
    return {
        'stat': stat,
        'value': _number(groups[-1]),
        'df': [_number(g) for g in groups[:-1] if g] if stat in WITH_DF else [],
        'relation': groups[0] if stat == 'p' else '=',
        'start': match.start(),
        'end': match.end(),
        'text': text[match.start():match.end()],
    }


def find_statistics(text):
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    clean = text.translate(MINUS_SIGNS)
    found = [_found(stat, match, text) for stat, pattern in PATTERNS.items()
             for match in re.finditer(pattern, clean)]
    return sorted(found, key=lambda item: item['start'])
