"""Treatment-control comparisons built from table and text records."""

import re
from itertools import accumulate, groupby

from .effectsize import allocate_n
from .fulltext import split_sentences
from .output import COLUMNS
from .statistics import find_statistics

ARM_STATS = ('mean', 'sd', 'n', 'se', 'adj_mean')
TEST_STATS = ('t', 'f', 'mse')
WHERE = ('pdf_page', 'printed_page', 'paragraph', 'table', 'section')
CLUSTER_UNITS = re.compile(r'(?i)\b(?:teachers?|classrooms?|classes|schools?|clusters?|sites?'
                           r'|districts?)\b')
OTHER_COUNT = re.compile(r'(?i)\s*\)?\s*(?:teachers?|classrooms?|classes|schools?|clusters?|sites?'
                         r'|districts?|withdr[ae]w\w*|dropped|left|attrit\w*|lost|excluded'
                         r'|were (?:excluded|lost|dropped))\b')
CODE = (r'(?:sld|ld|ebd|ed|ohi|mmd|mid|id|asd|adhd|iep|504|ell|esl|frl|swo?d|ba|bs|ma|ms|phd'
        r'|white|black|asian|boys?|girls?|speech|hearing|visual|grade|age|gender'
        r'|years?(?: of)? teaching|teaching experience)')
CHARACTERISTIC = re.compile(
    r'(?i)^\s*\d|\b(?:caucasian|african[- ]american|hispanic|latin[oax]s?|native american'
    r'|american indian|pacific islander|biracial|multiracial|race|ethnicity|gender|male|female'
    r'|lunch\w*|free or reduced|english (?:language )?learners?|disabilit\w*|special education'
    r'|autism|service area|degrees? earned|highest degree|years of experience'
    r'|grade\s*\d+|\d+(?:st|nd|rd|th) grade|age\s*\d+|\d+\s*years? old)\b'
    rf'|^\s*{CODE}(?:\s*[,/&]\s*{CODE})*\b[\s\d.,%()/-]*$')
NOT_ARM = re.compile(r'(?i)^\s*\(|^\s*(?:m|sd|se|n|mean|median|range|min|max|total|overall|sum'
                     r'|intercept|parameters?|estimates?|variables?|hlm models?|models?)'
                     r'\b[\s\d.,%()/=+-]*$')
CUE_WORDS = 3
TEXT_CUES = (('treatment', re.compile(r'(?i)\b(?:intervention|treatment|experimental)\b')),
             ('control', re.compile(r'(?i)\b(?:control|comparison|business[- ]as[- ]usual'
                                    r'|typical practice|wait[- ]?list)\b')))
NEAR_WORDS = 2
HEAD_WORDS = 3


def text_statistics(text, paper_id=None, outcome=None, time=None, where=None):
    """t, F, and MSE stated in ``text`` as records that ``build_comparisons`` can attach.

    Each record keeps its degrees of freedom in ``df``. ``outcome``, ``time``,
    and the location keys in ``where`` come from the caller.
    """
    place = {key: (where or {}).get(key) for key in WHERE}
    return [{**dict.fromkeys(COLUMNS), **place, 'paper_id': paper_id, 'field': found['stat'],
             'value': found['value'], 'layer': 'regex', 'status': 'extracted',
             'quote': found['text'], 'section': place['section'] or '', 'stat': found['stat'],
             'outcome': outcome, 'time': time, 'candidates': 0, 'df': found['df']}
            for found in find_statistics(text) if found['stat'] in TEST_STATS]


def _near(between):
    return len(re.findall(r'\w+', between)) <= NEAR_WORDS and not re.search(r'[.;?!]\s', between)


def _counts_clusters(text, start):
    head = re.split(r'[.;?!]\s', text[:start])[-1]
    return bool(CLUSTER_UNITS.search(' '.join(re.findall(r'\w+', head)[-HEAD_WORDS:])))


def _counts_other(text, end):
    """True when the word after an n value names clusters or attrition, as in 'n = 5 withdrew'."""
    return bool(OTHER_COUNT.match(text[end:]))


def text_arm_n(text, arms, paper_id=None, where=None):
    """n records for arms named just before an 'n = N' in ``text``.

    ``arms`` maps an arm label to its group. A run of labels followed by a run
    of n values, each at most two words from the one before, pairs the last
    labels with the values in order. A run headed by a cluster unit (teachers,
    classes, schools) within three words counts clusters and is skipped, as
    is a value followed by a cluster unit or an attrition word.
    """
    names = {label.lower(): label for label in arms or {}}
    if not names:
        return []
    pattern = '|'.join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    labels = [{'start': m.start(), 'end': m.end(), 'arm': names[m.group(0).lower()]}
              for m in re.finditer(rf'(?i)(?<!\w)(?:{pattern})(?!\w)', text)]
    events = sorted(labels + [s for s in find_statistics(text) if s['stat'] == 'n'],
                    key=lambda e: e['start'])
    breaks = [a is None or not _near(text[a['end']:b['start']]) or ('arm' in b and 'arm' not in a)
              for a, b in zip([None, *events], events)]
    blocks = [[e for _, e in group] for _, group in groupby(zip(accumulate(breaks), events),
                                                            key=lambda pair: pair[0])]
    place = {key: (where or {}).get(key) for key in WHERE}
    return [{**dict.fromkeys(COLUMNS), **place, 'paper_id': paper_id, 'field': 'n',
             'value': int(found['value']), 'layer': 'regex', 'status': 'extracted',
             'quote': ' '.join(text[block[0]['start']:block[-1]['end'] + text[block[-1]['end']:].startswith(')')].split()),
             'section': place['section'] or '', 'stat': 'n', 'arm': label['arm'],
             'group': arms[label['arm']], 'candidates': 0, 'df': []}
            for block in blocks
            for named, values in [([e for e in block if 'arm' in e],
                                   [e for e in block if 'arm' not in e
                                    and not _counts_other(text, e['end'])])]
            if values and len(values) <= len(named) and not _counts_clusters(text, block[0]['start'])
            for label, found in zip(named[-len(values):], values)]


def _near_label(sentence, name):
    """Words within ``CUE_WORDS`` of each mention of the label, the label itself left out."""
    return [' '.join(sentence[:m.start()].split()[-CUE_WORDS:] + sentence[m.end():].split()[:CUE_WORDS])
            for m in name.finditer(sentence)]


def _label(text):
    return re.compile(rf'(?i)(?<!\w){re.escape(text)}(?!\w)')


def _text_cue(arm, pages, peers=()):
    """Role the paper's sentences give an arm label, with the page and sentence as the reason.

    A cue word counts only within ``CUE_WORDS`` words of the label, and a
    sentence naming another arm of the same table (``peers``) describes the
    comparison, not this arm, so it is skipped.
    """
    name, others = _label(arm), [_label(peer) for peer in peers]
    found = [(role, page.get('page'), ' '.join(sentence.split()))
             for page in pages for _, sentence in split_sentences(page.get('text') or '')
             if not any(other.search(sentence) for other in others)
             for near in _near_label(sentence, name)
             for role, cue in TEXT_CUES if cue.search(near)]
    roles = {role for role, _, _ in found}
    if len(roles) != 1:
        return None
    role, page, sentence = found[0]
    return role, f'text p. {page}: {sentence[:90]}'


def suggest_roles(records, pages=None):
    """A suggested role and its reason for each table arm without a treatment or control group.

    Keyed by (paper_id, table, arm). Suggestions come from the data, for a
    coder to confirm or change: participant characteristics (race, grade,
    disability, lunch) and statistic, total, or model rows are 'other'; a
    label the paper's sentences tie to only treatment or only control words,
    each within three words of the label, in sentences naming no other arm of
    the table, takes that role; and the one open
    arm of a table whose one labeled arm is control (or treatment) takes the
    other role. An arm with no suggestion gets ``(None, '')``.
    """
    groups = {(r['paper_id'], r.get('table'), r['arm']): r.get('group') for r in records
              if r.get('table') is not None and r.get('arm')}
    open_keys = [key for key, group in groups.items() if group not in ('treatment', 'control')]
    found = {key: ('other', 'set to other at extraction') if groups[key] == 'other'
             else ('other', 'participant characteristic') if CHARACTERISTIC.search(key[2])
             else ('other', 'statistic, total, or model row') if NOT_ARM.search(key[2])
             else _text_cue(key[2], (pages or {}).get(key[0], ()),
                            [k[2] for k in groups if k[:2] == key[:2] and k != key]) or (None, '')
             for key in open_keys}
    settled = {key: group for key, group in groups.items() if group in ('treatment', 'control')}
    pending = {key[:2]: [k for k in open_keys if k[:2] == key[:2] and not found[k][0]]
               for key in open_keys}
    partner = {'control': 'treatment', 'treatment': 'control'}
    return {key: found[key] if found[key][0] else next(
                ((partner[role], f'other arm in the table is {role}')
                 for role in partner
                 if pending[key[:2]] == [key]
                 and [g for k, g in settled.items() if k[:2] == key[:2]] == [role]),
                (None, ''))
            for key in open_keys}


def text_total_n(text):
    """The total N when ``text`` states one value as 'N = ...', else None."""
    found = {s['value'] for s in find_statistics(text) if s['stat'] == 'n' and s['text'][0] == 'N'}
    return int(found.pop()) if len(found) == 1 else None


def _key(record):
    return (record['paper_id'], record['table'], record['outcome'], record['time'])


def _timed(record):
    """Arm values that carry a time point; an n without an outcome belongs to the arm label."""
    return bool(record.get('arm') and record.get('stat')
                and not (record['stat'] == 'n' and record.get('outcome') is None))


def _table_times(records):
    times = {}
    [times.setdefault((r.get('paper_id'), r.get('table')), set()).add(r.get('time'))
     for r in records if _timed(r)]
    return times


def _post_times(records, times):
    """Untimed arm values become posttest values in a table whose only stated time is post."""
    post = {table for table, found in times.items() if found == {None, 'post'}}
    return [{**r, 'time': 'post'} if _timed(r) and r.get('time') is None
            and (r.get('paper_id'), r.get('table')) in post else r for r in records]


def _one(records, stat, name, checks):
    found = list(dict.fromkeys(r['value'] for r in records if r['stat'] == stat))
    checks.extend([f'{name}_{stat}_duplicate'] if len(found) > 1 else [])
    return found[0] if len(found) == 1 else None


def _pre(arms, key, label):
    paper, table, outcome, _ = key
    rows = [r for r in arms if (r['paper_id'], r['outcome'], r['arm'], r['time'])
            == (paper, outcome, label, 'pre')]
    tables = list(dict.fromkeys(r['table'] for r in rows))
    chosen = table if table in tables else tables[0] if len(tables) == 1 else None
    return [r for r in rows if r['table'] == chosen]


def _arm(arms, key, label, role, checks):
    post = [r for r in arms if _key(r) == key and r['arm'] == label]
    header = [r for r in arms if (r['paper_id'], r['table'], r['arm']) == (key[0], key[1], label)
              and r['stat'] == 'n' and r['outcome'] is None and r['time'] is None]
    pre = _pre(arms, key, label)
    stats = {'label': label,
             'post': {stat: _one(post, stat, f'{role}_post', checks)
                      for stat in ARM_STATS if stat != 'n'},
             'pre': {stat: _one(pre, stat, f'{role}_pre', checks) for stat in ARM_STATS}}
    stats['post']['n'] = (_one(post, 'n', f'{role}_post', checks)
                          if any(r['stat'] == 'n' for r in post)
                          else _one(header, 'n', f'{role}_header', checks))
    return stats, [r['quote'] for r in post + header + pre]


def _allocate(stats, labels, stated, study, routes, checks):
    """Fill missing n by proportional allocation of the study N when no arm states one."""
    missing = [role for role in ('treatment', 'control') if stats[role]['post']['n'] is None]
    total = study.get('n_total')
    if not missing:
        return
    if stated or total is None:
        checks.append('n_missing')
        return
    weights = study.get('allocation') or {}
    shares = dict(zip(labels, allocate_n(total, [weights.get(label, 1) for label in labels])))
    [stats[role]['post'].update({'n': shares[stats[role]['label']]}) for role in missing]
    routes.append('n_allocated')
    checks.extend(['n_allocation_rounded'] if sum(shares.values()) != total else [])


def _sd_missing(absent, key, stats):
    """Arms without a posttest SD whose SD (or, lacking an SE too, SE) is empty or unreadable."""
    lacking = {arm['label'] for arm in stats.values() if arm['post'].get('sd') is None}
    bare = {arm['label'] for arm in stats.values() if arm['post'].get('se') is None} & lacking
    return list(dict.fromkeys(f"sd_missing: {r['arm']} ({r['stat']})" for r in absent
                              if _key(r) == key and (r['arm'] in lacking
                                                     and r['stat'] in ('sd', 'mean_sd')
                                                     or r['arm'] in bare and r['stat'] == 'se')))


def _matches(test, key):
    paper, table, outcome, time = key
    return (test['paper_id'] == paper and test['table'] in (None, table)
            and test['outcome'] in (None, outcome) and test['time'] in (None, time))


def _pairs(arms, key, pairing):
    labels = {role: list(dict.fromkeys(r['arm'] for r in arms
                                       if _key(r) == key and r['group'] == role))
              for role in ('treatment', 'control')}
    chosen = (pairing or {}).get(key[:2]) or {}
    elsewhere = any(r['group'] == 'control' and _key(r)[:3] == key[:3]
                    and r['time'] not in ('pre', key[3]) for r in arms)
    controls = labels['control']
    rule = (controls[0] if len(controls) == 1 else None,
            ['several_controls'] if len(controls) > 1
            else ['control_at_other_time'] if not controls and elsewhere
            else ['missing_control'] if not controls else [])
    human = {label: (control, [] if control in controls else ['override_control_not_found'])
             for label, control in chosen.items()}
    return [(label, *(human[label] if label in human else rule),
             'human' if label in human else 'rule')
            for label in labels['treatment'] if not (label in chosen and chosen[label] is None)]


def _implausible(record):
    return (record['stat'] in ('f', 'mse') and record['value'] < 0
            or min(record.get('df') or [1]) <= 0)


def _stub(key, check, quotes, arm=None):
    paper, table, outcome, time = key
    return {'paper_id': paper, 'table': table, 'outcome': outcome, 'time': time,
            'es_id': ':'.join(str(part) for part in (*key, arm) if part is not None) if arm
            else None,
            'treatment': None, 'control': None, 'control_id': None,
            'shared_control': False, 'pairing': None, 'study': {}, 'arms': {}, 'tests': {},
            'inputs': {}, 'routes': [], 'checks': [check], 'quotes': quotes}


def _comparison(context, key, pair, study):
    arms = context['arms']
    treatment, control, checks, source = pair
    checks, routes = list(checks), []
    stats_t, quotes_t = _arm(arms, key, treatment, 'treatment', checks)
    stats_c, quotes_c = (_arm(arms, key, control, 'control', checks) if control
                         else ({'label': None, 'post': {'n': None}, 'pre': {}}, []))
    stats = {'treatment': stats_t, 'control': stats_c}
    lent = [role for role in stats if stats[role]['post'].get('n') is None
            and stats[role]['pre'].get('n') is not None]
    [stats[role]['post'].update({'n': stats[role]['pre']['n']}) for role in lent]
    routes.extend(['n_from_pretest'] if lent else [])
    paper, table_id, outcome, time = key
    other = {role: [r for r in arms if r['stat'] == 'n' and r['paper_id'] == paper
                    and r['table'] != table_id and r['arm'] == stats[role]['label']
                    and r['outcome'] in (None, outcome) and r['time'] in (None, 'post', time)]
             for role in stats if stats[role]['post'].get('n') is None}
    other = {role: [r for r in rows if r['outcome'] == outcome] or rows
             for role, rows in other.items()}
    borrowed = {role: _one(rows, 'n', f'{role}_other_source', checks)
                for role, rows in other.items() if rows}
    [stats[role]['post'].update({'n': n}) for role, n in borrowed.items() if n is not None]
    found_n = ['n_from_other_source'] if any(n is not None for n in borrowed.values()) else []
    routes.extend(found_n)
    checks.extend(found_n)
    quotes_t += [r['quote'] for rows in other.values() for r in rows] if found_n else []
    table = [r for r in arms if (r['paper_id'], r['table']) == key[:2]]
    if control:
        _allocate(stats, list(dict.fromkeys(r['arm'] for r in table)),
                  any(r['stat'] == 'n' for r in table), study, routes, checks)
    found = {stat: [r for r in context['tests'] if r['stat'] == stat and _matches(r, key)]
             for stat in TEST_STATS}
    shared = [r for r in context['shared'] if _matches(r, key)]
    checks.extend(f'{stat}_ambiguous' for stat, rows in found.items() if len(rows) > 1)
    checks.extend(['statistic_shared'] if shared else [])
    checks.extend(['statistic_implausible']
                  if any(_implausible(r) for rows in found.values() for r in rows) else [])
    checks.extend(_sd_missing(context['absent'], key, stats))
    checks.extend(['time_unclear'] if key[3] is None and key[:2] in context['unclear'] else [])
    es_id = ':'.join(str(part) for part in (*key, treatment) if part is not None)
    return {'paper_id': paper, 'table': table_id, 'outcome': outcome, 'time': time,
            'es_id': es_id,
            'treatment': treatment, 'control': control,
            'control_id': ':'.join(str(part) for part in (paper, table_id, control)) if control
            else None,
            'pairing': source, 'study': study,
            'inputs': (context['inputs'] or {}).get(es_id) or {},
            'arms': stats,
            'tests': {stat: {'value': rows[0]['value'], 'df': rows[0].get('df') or []}
                      for stat, rows in found.items() if len(rows) == 1},
            'routes': routes, 'checks': checks,
            'quotes': quotes_t + quotes_c + [r['quote'] for rows in found.values() for r in rows]
            + [r['quote'] for r in shared]}


def build_comparisons(records, studies=None, pairing=None, inputs=None):
    """One comparison per treatment arm, outcome, and time point, paired with its control.

    ``records`` come from ``extract_table_values``, ``text_statistics``, or a
    coder, and ``extracted`` and ``review`` values are used (a comparison resting on
    a ``review`` value gets ``values_review``), never ``not_found`` or ``refused``; an empty list raises
    ``ValueError``. Arms pair within a paper, table, outcome, and time point.
    In a table whose only stated time is post, untimed values are posttest
    values; beside a pretest or follow-up they stay untimed and are flagged
    ``time_unclear``. Pretest values of the same arm and outcome join the
    comparison. An arm with no posttest or pretest n takes one from another table or
    text record of the paper with the same arm label and outcome, else with no outcome
    (as ``text_arm_n`` gives), with route and check ``n_from_other_source``.
    ``studies`` maps a paper to ``design`` (rct, qed),
    ``assignment`` (individual, cluster), ``clusters``, ``cluster_corrected``,
    ``n_total``, ``allocation`` weights by arm, ``rho``, ``icc``, and ``r2``.
    ``pairing`` maps ``(paper_id, table)`` to ``{treatment arm: control arm}``; it always
    wins, and ``None`` drops the arm. ``inputs`` maps an ``es_id`` to
    per-comparison values: ``domain`` (achievement, behavioural, other) and a
    ``var_g`` override; a key matching no ``es_id`` raises ``ValueError``. A t, F, or MSE joins the one comparison it matches; one
    matching several joins none, and each of them gets ``statistic_shared``.
    Arms without a settled role, test statistics that match no comparison,
    and a paper with no comparison come back as comparisons with only a check.
    """
    if not records:
        raise ValueError('build_comparisons needs at least one record')
    times = _table_times(records)
    records = _post_times(records, times)
    unclear = {table for table, found in times.items()
               if None in found and found & {'pre', 'follow_up'}}
    used = [r for r in records if r.get('stat') and (r.get('status') == 'extracted' or (
        r.get('status') == 'review' and r.get('value') is not None))]
    flagged = {r['quote'] for r in used if r['status'] == 'review'}
    unconfirmed = {(r.get('paper_id'), r.get('table')) for r in records
                   if r.get('status') == 'review' and r.get('stat')
                   and r.get('value') is None}
    absent = [r for r in records if r.get('status') in ('not_found', 'review')
              and r.get('stat') and r.get('arm')]
    arms = [r for r in used if r.get('group') in ('treatment', 'control', 'other') and r.get('arm')]
    tests = [r for r in used if r['stat'] in TEST_STATS and not r.get('arm')]
    treated = [r for r in arms if r['group'] == 'treatment' and r['time'] != 'pre'
               and not (r['stat'] == 'n' and r['table'] is None and r['outcome'] is None)]
    valued = {_key(r)[:2] for r in treated if r['stat'] != 'n'}
    keys = list(dict.fromkeys(_key(r) for r in treated
                              if r['stat'] != 'n' or _key(r)[:2] not in valued))
    pairs = [(key, pair) for key in keys for pair in _pairs(arms, key, pairing)]
    hits = [sum(_matches(r, key) for key, _ in pairs) for r in tests]
    context = {'arms': arms, 'tests': [r for r, hit in zip(tests, hits) if hit == 1],
               'shared': [r for r, hit in zip(tests, hits) if hit > 1], 'absent': absent,
               'unclear': unclear, 'inputs': inputs}
    comparisons = [_comparison(context, key, pair, (studies or {}).get(key[0]) or {})
                   for key, pair in pairs]
    known = {c['es_id'] for c in comparisons if c['es_id']}
    unknown = [key for key in (inputs or {}) if key not in known]
    if unknown:
        raise ValueError(f'inputs keys match no es_id: {unknown}')
    [c['checks'].append('values_review') for c in comparisons
     if flagged & set(c['quotes'])]
    suggested = {r['quote'] for r in used if r.get('group_source') == 'suggested'}
    [c['checks'].append('role_suggested') for c in comparisons if suggested & set(c['quotes'])]
    [c['checks'].append('unconfirmed_values') for c in comparisons
     if (c['paper_id'], c['table']) in unconfirmed]
    unsettled = list(dict.fromkeys((_key(r), r['arm']) for r in used
                                   if r.get('arm') and r.get('group') is None))
    comparisons += [_stub(key, 'arm_role_unsettled',
                          [r['quote'] for r in used if _key(r) == key and r.get('arm') == arm],
                          arm)
                    for key, arm in unsettled]
    comparisons += [_stub(_key(r), 'test_statistic_unpaired', [r['quote']])
                    for r, hit in zip(tests, hits) if not hit]
    covered = {c['paper_id'] for c in comparisons}
    comparisons += [_stub((paper, None, None, None), 'no_comparisons', [])
                    for paper in dict.fromkeys(r.get('paper_id') for r in records)
                    if paper not in covered]
    shared = {}
    [shared.setdefault((c['control_id'], c['outcome'], c['time']), []).append(c)
     for c in comparisons if c['control_id']]
    [c.update({'shared_control': len(shared.get((c['control_id'], c['outcome'], c['time']), []))
               > 1}) for c in comparisons]
    return comparisons
