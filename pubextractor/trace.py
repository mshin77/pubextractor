"""Per-paper trace graphs of extracted values, coder decisions, and PRISMA methods text."""

import json
from collections import Counter
from pathlib import Path

from .provenance import log_human

STATUS = {'extracted': 'confirmed', 'review': 'review', 'refused': 'refused',
          'not_found': 'missing'}
RANK = ('refused', 'review', 'missing', 'confirmed')
KEYS = ('paper_id', 'field')
ACTIONS = {'confirm': 'extracted', 'correct': 'extracted', 'not_reported': 'not_found'}
CHOICE = {'regex': 'rule', 'table': 'rule', 'prompt': 'model choice',
          'vision': 'model choice', 'human': 'coder', 'none': 'none'}
WHERE = ('pdf_page', 'printed_page', 'section', 'table', 'row', 'col')
LAYER_NAMES = {'regex': 'text rules', 'table': 'table parsing', 'prompt': 'a language model',
               'vision': 'a vision model', 'human': 'coder entry'}
DONE = {'confirm': 'confirmed', 'correct': 'corrected', 'not_reported': 'marked not reported',
        'role': 'arm roles set', 'calibrate': 'graphs calibrated', 'enter': 'tables entered'}


def _chain(graph, start, nodes):
    ids = [start] + [node['id'] for node in nodes]
    graph['nodes'] += nodes
    graph['edges'] += [{'from': a, 'to': b} for a, b in zip(ids, ids[1:])]


def _feed(graph, counts, prefix, label, sink, sink_label):
    graph['nodes'] += ([{'id': f'{prefix}:{key}', 'level': 'derived', 'label': f'{label} {key}',
                         'value': count} for key, count in counts.items()]
                       + [{'id': sink, 'level': 'derived', 'label': sink_label,
                           'value': sum(counts.values())}] * bool(counts))
    graph['edges'] += [edge for key in counts for edge in (
        {'from': 'study', 'to': f'{prefix}:{key}'}, {'from': f'{prefix}:{key}', 'to': sink})]


def _count(n, word):
    return f"{n} {word}{'s' * (n != 1)}"


def trace_graph(records, effects=(), graph_rows=(), tracker=None):
    """One graph per paper: ``nodes``, ``edges``, ``evidence``, and ``status`` per field.

    Each value runs study -> field -> source -> layer -> candidates -> choice ->
    checks -> imputation -> export, then the human steps ``tracker`` holds for
    that field. Table values feed g and v_g, graph points feed scdbayes rows,
    and correlation matrices feed matrix files. Field status is the worst of its
    values: refused, review, missing, then confirmed.
    """
    meta = tracker.to_dict() if tracker is not None else {}
    papers = dict.fromkeys([r['paper_id'] for r in records] + [e['paper_id'] for e in effects]
                           + [row.get('study') for row in graph_rows])
    graphs = {}
    for paper in papers:
        graph = graphs[paper] = {'nodes': [{'id': 'study', 'level': 'study', 'label': paper}],
                                 'edges': [], 'evidence': {}, 'status': {}}
        license_verdict = meta.get('permissions', {}).get(paper, {}).get('verdict')
        fields = [r for r in records if r['paper_id'] == paper and 'field' in r]
        for i, r in enumerate(fields):
            graph['status'][r['field']] = min(graph['status'].get(r['field'], 'confirmed'),
                                              STATUS.get(r['status'], 'review'), key=RANK.index)
            where = {key: r.get(key) for key in WHERE}
            steps = {'source': where, 'layer': r['layer'], 'candidates': r.get('candidates', 0),
                     'choice': CHOICE.get(r['layer'], r['layer']), 'checks': r['status'],
                     'imputation': None, 'export': r['value']}
            human = [h for h in meta.get('human', []) if (h['paper_id'], h['field'])
                     == (paper, r['field'])]
            _chain(graph, f"field:{r['field']}",
                   [{'id': f'v{i}.{step}', 'level': 'value', 'field': r['field'], 'step': step,
                     'value': value} for step, value in steps.items()]
                   + [{'id': f'v{i}.human{k}', 'level': 'value', 'field': r['field'],
                       'step': 'human', 'value': h['after'], 'action': h['action'],
                       'coder': h['coder'], 'time': h['time']} for k, h in enumerate(human)])
            graph['evidence'][f'v{i}'] = {'quote': r.get('quote'), **where, 'license': license_verdict}
        graph['nodes'] += [{'id': f'field:{field}', 'level': 'field', 'label': field,
                            'status': status} for field, status in graph['status'].items()]
        graph['edges'] += [{'from': 'study', 'to': f'field:{field}'} for field in graph['status']]
        for k, e in enumerate(e for e in effects if e['paper_id'] == paper):
            graph['nodes'].append({'id': f'g{k}', 'level': 'derived', 'label': 'g',
                                   'value': e.get('g'), 'route': e.get('imputed_route'),
                                   'method': e.get('method'), 'status': e.get('status')})
            graph['edges'] += [{'from': f'v{i}.export', 'to': f'g{k}'}
                               for i, r in enumerate(fields) if r.get('stat')
                               and (r.get('table'), r.get('outcome')) == (e['table'], e['outcome'])
                               and r.get('arm') in (e['treatment'], e['control'])]
            _chain(graph, f'g{k}', [{'id': f'vg{k}', 'level': 'derived', 'label': 'v_g',
                                     'value': e.get('var_g')}])
        _feed(graph, Counter(str(row.get('figure')) for row in graph_rows
                             if row.get('study') == paper),
              'points', 'Figure', 'scdbayes', 'scdbayes rows')
        _feed(graph, Counter(r.get('table') for r in records
                             if r['paper_id'] == paper and 'var_i' in r),
              'matrix', 'Table', 'matrices', 'Matrix files')
    return graphs


def write_trace(graphs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graphs, indent=2, default=str), encoding='utf-8')
    return path


def read_decisions(path):
    """Coder decisions: a list of {paper_id, field, action, value, coder, time}.

    ``action`` is confirm, correct, or not_reported; any other, a file that is
    not a list of objects, or a decision without ``paper_id`` or ``field``
    raises ``ValueError``.
    """
    decisions = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(decisions, list):
        raise ValueError(f"decisions must be a list, not a JSON {type(decisions).__name__}")
    shape = [f"decision {i}: " + ('missing ' + ', '.join(k for k in KEYS if k not in d)
                                  if isinstance(d, dict) else 'not an object')
             for i, d in enumerate(decisions, start=1)
             if not isinstance(d, dict) or not set(KEYS) <= set(d)]
    if shape:
        raise ValueError('; '.join(shape))
    unknown = [str(d.get('action')) for d in decisions if d.get('action') not in ACTIONS]
    if unknown:
        raise ValueError(f"unknown action {', '.join(unknown)}: use confirm, correct, or "
                         'not_reported')
    return decisions


def apply_decisions(records, decisions, tracker=None):
    """Records with each decision applied as layer ``human``, and the human steps logged.

    A decision applies to every record of its paper and field; one matching no
    record is skipped.
    """
    keyed = {(d['paper_id'], d['field']): d for d in decisions}
    value = {'confirm': lambda r, d: r['value'], 'correct': lambda r, d: d.get('value'),
             'not_reported': lambda r, d: None}
    updated = [{**r, 'layer': 'human', 'status': ACTIONS[d['action']],
                'value': value[d['action']](r, d)}
               if (d := keyed.get((r['paper_id'], r.get('field')))) else r for r in records]
    before = {(r['paper_id'], r.get('field')): r['value'] for r in records if 'field' in r}
    after = {(r['paper_id'], r.get('field')): r['value'] for r in updated if 'field' in r}
    steps = [log_human(tracker, *key, d['action'], before[key], after[key], d.get('coder'),
                       d.get('time')) for key, d in keyed.items() if key in before]
    return updated, steps


def prisma_items(tracker, records):
    """Editable methods text for PRISMA 2020 items 9 and 11, from the provenance."""
    from . import __version__

    meta = tracker.to_dict() if tracker is not None else {}
    fields = [r for r in records if 'field' in r]
    layers = Counter(r['layer'] for r in fields)
    states = Counter(r['status'] for r in fields)
    prompts = meta.get('prompts', {})
    models = list(dict.fromkeys(p['model'] for p in prompts.get('records', []) if p['model']))
    verdicts = Counter(p.get('verdict') for p in meta.get('permissions', {}).values())
    human = Counter(h['action'] for h in meta.get('human', []))
    papers = len(dict.fromkeys(r['paper_id'] for r in records))
    tool = f'pubextractor {__version__}'
    item_9 = ' '.join([
        f"Data were extracted with {tool} from {_count(papers, 'paper')}: "
        + ', '.join([_count(len(fields), 'value')]
                    + [f'{n} by {LAYER_NAMES.get(layer, layer)}'
                       for layer, n in layers.most_common() if layer != 'none'])
        + '.',
        f"{_count(prompts['count'], 'prompt')} went to {', '.join(models) or 'an unnamed model'}"
        '; each prompt is logged by SHA-256 hash with its model name.' if prompts
        else 'No language model was used.',
        'License verdicts, one per paper, before any model call: '
        + ', '.join(f'{n} {verdict}' for verdict, n in verdicts.items()) + '.' if verdicts
        else 'No license verdict was recorded.',
        f"Human verification: {_count(sum(human.values()), 'step')} ("
        + ', '.join(f'{n} {DONE.get(action, action)}' for action, n in human.items())
        + '), each logged with coder and time.' if human
        else 'No human verification step was recorded.',
        f"At export, {_count(states['review'], 'value')} remained flagged for review, "
        f"{states['not_found']} not found, and {states['refused']} refused."])
    item_11 = (f"Automation tools: {tool} for data extraction"
               + (f", with {', '.join(models)}. A model could choose only a value from the "
                  "field's closed list or a number within its range; any other answer was "
                  'flagged for review' if models else ', by rules only')
               + '. No automation tool made a risk-of-bias judgment. Each value keeps the page '
               'and quote it rests on.')
    return {'item 9': item_9, 'item 11': item_11}
