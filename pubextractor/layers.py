"""Extract codebook fields from a document: rules first, then a model where allowed."""

import json
import re
from itertools import accumulate

from .codebook import _is_number
from .confidence import quote_found
from .fulltext import _heading_name, section_at, split_sentences
from .highlight import pattern_from_terms
from .rag import _check_permission, extract_with_rag
from .statistics import find_statistics

NO_TEXT = 'no extractable text (scanned or empty file)'


def _page_text(page):
    rows = page.get('rows')
    table = '\n'.join(' | '.join(row) for row in rows) if rows else ''
    return page.get('text') or table


def _heading_lines(page, text):
    marks = {offset: name.lower() for offset, name in page.get('section_marks', [])}
    lines = text.splitlines(keepends=True)
    return [(start, start + len(line))
            for start, line in zip(accumulate(map(len, lines), initial=0), lines)
            if _heading_name(line) or (start in marks and marks[start]
                                       and line.strip().lower().endswith(marks[start]))]


def _segments(text, headings):
    bounds = [0, *[edge for heading in headings for edge in heading], len(text)]
    return [(bounds[i], text[bounds[i]:bounds[i + 1]]) for i in range(0, len(bounds), 2)]


def _sentences(pages):
    return [{'text': sentence.strip(),
             'start': base + start + len(sentence) - len(sentence.lstrip()), 'page': page}
            for page in pages
            for text in [page.get('text') or '']
            for base, segment in _segments(text, _heading_lines(page, text))
            for start, sentence in split_sentences(segment.replace('\n', ' '))
            if sentence.strip()]


def _where(sentence, offset):
    page = sentence['page']
    return {'text': sentence['text'], 'pdf_page': page.get('page'),
            'printed_page': page.get('printed_page'),
            'paragraph': page.get('paragraph'), 'table': page.get('table'),
            'section': section_at(page, sentence['start'] + offset)}


def _record(paper_id, name, value, layer, status, where=None, candidates=0, quote=None):
    where = where or {}
    return {'paper_id': paper_id, 'field': name, 'value': value, 'layer': layer,
            'status': status, 'quote': where.get('text', '') if quote is None else quote,
            'pdf_page': where.get('pdf_page'), 'printed_page': where.get('printed_page'),
            'paragraph': where.get('paragraph'), 'table': where.get('table'),
            'row': None, 'col': None, 'section': where.get('section', ''),
            'group': None, 'arm': None, 'time': None, 'stat': None, 'outcome': None,
            'candidates': candidates}


def _in_range(field, value):
    low, high = field['range'] or (float('-inf'), float('inf'))
    whole = field['type'] != 'integer' or float(value).is_integer()
    return low <= value <= high and whole


def _rule_category(name, field, sentences, paper_id):
    patterns = {value: re.compile(pattern_from_terms(terms), re.IGNORECASE)
                for value, terms in field['terms'].items() if terms}
    hits = [(value, sentence, match) for sentence in sentences
            for value, pattern in patterns.items()
            for match in [pattern.search(sentence['text'])] if match]
    values = list(dict.fromkeys(value for value, _, _ in hits))
    if not values:
        return None
    first = {value: next(_where(s, m.start()) for v, s, m in hits if v == value)
             for value in values}
    if field['type'] == 'multi_category':
        return [_record(paper_id, name, value, 'regex', 'extracted', first[value],
                        candidates=len(values)) for value in values]
    status = 'extracted' if len(values) == 1 else 'review'
    return [_record(paper_id, name, values[0], 'regex', status, first[values[0]],
                    candidates=len(values))]


def _rule_number(name, field, sentences, paper_id):
    found = [(stat, sentence) for sentence in sentences
             for stat in find_statistics(sentence['text'])
             if stat['stat'] == field['stat'] and _in_range(field, stat['value'])]
    if not found:
        return None
    stat, sentence = found[0]
    value = int(stat['value']) if field['type'] == 'integer' else stat['value']
    status = 'extracted' if len(found) == 1 else 'review'
    return [_record(paper_id, name, value, 'regex', status, _where(sentence, stat['start']),
                    candidates=len(found))]


RULES = {'category': _rule_category, 'multi_category': _rule_category,
         'integer': _rule_number, 'number': _rule_number}


CHOICE = ('Answer using only the passages below.\n\nQuestion: {query}\n\nPassages:\n{passages}\n\n'
          'Reply with JSON only: {{"value": <one of CHOICES, or null if not reported>, '
          '"quote": "<the sentence the answer rests on, copied exactly>", '
          '"passage": <passage number>}}')

ANSWER_KIND = {'integer': 'a number', 'number': 'a number', 'text': 'a short text'}


def _parse(answer):
    match = re.search(r'\{.*\}', answer or '', re.DOTALL)
    try:
        reply = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}
    return reply if isinstance(reply, dict) else {}


def _valid(field, value):
    number = _is_number(value)
    checks = {
        'category': lambda: isinstance(value, str) and value in field['values'],
        'multi_category': lambda: (isinstance(value, list) and bool(value)
                                   and all(isinstance(v, str) and v in field['values']
                                           for v in value)),
        'integer': lambda: number and _in_range(field, value),
        'number': lambda: number and _in_range(field, value),
        'text': lambda: isinstance(value, str) and bool(value.strip()),
    }
    return value is not None and checks[field['type']]()


def _listed(field, item):
    listed = {str(value).strip().lower(): value for value in field['values']}
    return listed.get(item.strip().lower(), item) if isinstance(item, str) else item


def _canonical(field, value):
    if isinstance(value, list):
        return [_listed(field, item) for item in value]
    return _listed(field, value)


def _prompt(name, field, pages, paper_id, model):
    choices = json.dumps(field['values']) if field['values'] else ANSWER_KIND[field['type']]
    template = CHOICE.replace('CHOICES', choices.replace('{', '{{').replace('}', '}}'))
    question = field['query'] or field['definition'] or name
    result = extract_with_rag(pages, {name: question}, model['respond'], model['encoder'],
                              template=template, tracker=model['tracker'],
                              permission=model['permission'], endpoint=model['endpoint'],
                              paper_id=paper_id, cache=model['cache'])[name]
    if result['error']:
        return [_record(paper_id, name, None, 'prompt', 'review',
                        quote=f"model call failed: {result['error']}")]
    reply = _parse(result['answer'])
    passages = result['passages']
    index = reply.get('passage')
    citation = (isinstance(index, int) and not isinstance(index, bool)
                and 1 <= index <= len(passages))
    cited = passages[index - 1] if citation else {}
    where = {'text': cited.get('text', ''), 'pdf_page': cited.get('pdf_page'),
             'printed_page': cited.get('printed_page'), 'paragraph': cited.get('paragraph'),
             'table': cited.get('table'), 'section': cited.get('section', '')}
    quote = reply.get('quote') if isinstance(reply.get('quote'), str) else ''
    value = _canonical(field, reply.get('value'))
    if 'value' in reply and value is None:
        return [_record(paper_id, name, None, 'prompt', 'not_found')]
    valid = _valid(field, value)
    status = 'extracted' if valid and quote_found(quote, [where['text']]) else 'review'
    if not valid:
        return [_record(paper_id, name, None, 'prompt', status, where, quote=quote)]
    values = (value if field['type'] == 'multi_category'
              else [int(value) if field['type'] == 'integer' else value])
    return [_record(paper_id, name, item, 'prompt', status, where, quote=quote)
            for item in values]


def _no_rule(*args):
    return None


def _fallback(name, field, pages, paper_id, model):
    if model['respond'] is None or model['encoder'] is None:
        return [_record(paper_id, name, None, 'none', 'not_found')]
    try:
        _check_permission(model['permission'], model['endpoint'])
    except (PermissionError, ValueError) as refusal:
        return [_record(paper_id, name, None, 'prompt', 'refused', quote=str(refusal))]
    try:
        return _prompt(name, field, pages, paper_id, model)
    except PermissionError as refusal:
        return [_record(paper_id, name, None, 'prompt', 'refused', quote=str(refusal))]
    except ValueError as failure:
        return [_record(paper_id, name, None, 'prompt', 'review',
                        quote=f'model step failed: {failure}')]


def extract_values(pages, codebook, paper_id=None, respond=None, encoder=None,
                   permission=None, endpoint=None, tracker=None):
    """One record per extracted value, rules first.

    Pages are the ``read_pdf`` page list; every record keeps the page and
    section of the text it rests on.
    """
    if not codebook:
        raise ValueError('codebook has no fields')
    pages = [{**page, 'text': _page_text(page)} for page in pages]
    if not any(page['text'].strip() for page in pages):
        return [_record(paper_id, name, None, 'none', 'not_found', quote=NO_TEXT)
                for name in codebook]
    sentences = _sentences(pages)
    model = {'respond': respond, 'encoder': encoder, 'permission': permission,
             'endpoint': endpoint, 'tracker': tracker, 'cache': {}}
    return [record for name, field in codebook.items()
            for record in (RULES.get(field['type'], _no_rule)(name, field, sentences, paper_id)
                           or _fallback(name, field, pages, paper_id, model))]
