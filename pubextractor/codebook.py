"""Codebook fields that say what to extract from each paper and how to check it."""
import os
import re

import yaml

TYPES = {'category', 'multi_category', 'integer', 'number', 'text'}
STATS = {'n', 'mean', 'sd', 'se', 't', 'f', 'r', 'p', 'mse'}
COLOR = re.compile(r'#(?:[0-9a-fA-F]{3}){1,2}')


def safe_color(value, default):
    return value if isinstance(value, str) and COLOR.fullmatch(value) else default


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_scalar(value):
    return isinstance(value, (str, int, float, bool))


def _text(value):
    return ('yes' if value else 'no') if isinstance(value, bool) else str(value)


def _problems(name, field):
    if not isinstance(field, dict):
        return [f"{name}: field must be a mapping"]
    kind = field.get('type')
    values = field.get('values') or []
    terms = field.get('terms') or {}
    span = field.get('range')
    stat = field.get('stat')
    color = field.get('color')
    valid_kind = isinstance(kind, str) and kind in TYPES
    values_ok = isinstance(values, (list, tuple))
    bad_values = [value for value in values if not _is_scalar(value)] if values_ok else []
    terms_ok = isinstance(terms, dict) and all(
        isinstance(found, (list, tuple)) and all(isinstance(term, str) for term in found)
        for found in terms.values())
    span_ok = span is None or (isinstance(span, (list, tuple)) and len(span) == 2
                               and all(map(_is_number, span)) and span[0] <= span[1])
    stat_ok = stat is None or (isinstance(stat, str) and stat in STATS)
    color_ok = color is None or (isinstance(color, str) and bool(COLOR.fullmatch(color)))
    checks = [
        (not valid_kind, f"{name}: type must be one of {sorted(TYPES)}, not {kind!r}"),
        (valid_kind and kind in ('category', 'multi_category') and values_ok and not values,
         f"{name}: values are required for {kind}"),
        (not values_ok, f"{name}: values must be a list"),
        (bool(bad_values), f"{name}: values must be text, not {bad_values[0]!r}" if bad_values else ''),
        (not stat_ok, f"{name}: stat must be one of {sorted(STATS)}, not {stat!r}"),
        (not span_ok, f"{name}: range must be [low, high] with numbers, low <= high"),
        (not color_ok, f"{name}: color must be a hex color such as '#1f77b4', not {color!r}"),
        (not terms_ok, f"{name}: terms must map values to lists of terms"),
        (values_ok and terms_ok and not bad_values
         and any(not _is_scalar(key) or _text(key) not in map(_text, values) for key in terms),
         f"{name}: terms name values not in values"),
    ]
    return [message for bad, message in checks if bad]


def _read_yaml(path):
    with open(path, encoding='utf-8-sig') as handle:
        loaded = yaml.safe_load(handle)
    return {} if loaded is None else loaded


def _normalize(field):
    return {
        'type': field['type'],
        'values': [_text(value) for value in field.get('values') or []],
        'definition': field.get('definition', ''),
        'terms': {_text(value): list(terms) for value, terms in (field.get('terms') or {}).items()},
        'stat': field.get('stat'),
        'range': tuple(field['range']) if field.get('range') is not None else None,
        'query': field.get('query'),
        'note': field.get('note'),
        'color': field.get('color'),
    }


def _load_raw(source, caller):
    is_path = isinstance(source, (str, os.PathLike))
    if is_path and not os.path.isfile(source):
        raise FileNotFoundError(f"{source!r} is not a file; {caller} takes a dict or a YAML file path")
    raw = _read_yaml(source) if is_path else source
    if not isinstance(raw, dict):
        raise ValueError('Codebook must be a mapping of field names to fields')
    return raw


def load_codebook(source):
    """Fields from a dict or a YAML file, checked and normalized.

    Raises ``ValueError`` naming every problem at once, so a codebook can be
    fixed in one pass.
    """
    raw = _load_raw(source, 'load_codebook')
    if not raw:
        raise ValueError('codebook has no fields')
    problems = [message for name, field in raw.items() for message in _problems(name, field)]
    if problems:
        raise ValueError('Codebook problems: ' + '; '.join(problems))
    return {name: _normalize(field) for name, field in raw.items()}


SCHEMA_TYPES = {'binary': 'category', 'categorical': 'category',
                'categorical_multiple': 'multi_category', 'number': 'number', 'text': 'text'}


def _from_schema(entry):
    return {
        'type': SCHEMA_TYPES[entry['type']],
        'values': ['yes', 'no'] if entry['type'] == 'binary' else list(entry.get('values') or []),
        'definition': entry.get('description', ''),
        'query': entry.get('query'),
        'note': entry.get('stage1_alignment'),
    }


def import_extraction_schema(source):
    """Codebook from the ``extraction_schema.yml`` layout.

    Returns the codebook and the names of entries it could not use, such as
    derived variables computed after extraction.
    """
    raw = _load_raw(source, 'import_extraction_schema')
    usable = {name: entry for name, entry in raw.items()
              if isinstance(entry, dict) and isinstance(entry.get('type'), str)
              and entry['type'] in SCHEMA_TYPES}
    skipped = sorted(set(raw) - set(usable))
    return load_codebook({name: _from_schema(entry) for name, entry in usable.items()}), skipped
