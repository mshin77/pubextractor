"""Study design suggested per paper from its Method text, tables, and figure captions."""

import re

from .figures import CAPTION_TYPES

DESIGN_TYPES = ('group', 'single_case', 'correlational', 'qualitative')
SINGLE_CASE_TERMS = {'multiple baseline': r'multiple[- ]baseline',
                     'multiple probe': r'multiple[- ]probe',
                     'single-case': r'single[- ]case', 'single-subject': r'single[- ]subject',
                     'reversal design': r'reversal design', 'ABAB': r'\bA-?B-?A-?B\b',
                     'alternating treatment': r'alternating[- ]treatments?'}
GROUP_TERMS = {'randomly assigned': r'randomly assigned',
               'random assignment': r'random assignment',
               'comparison group': r'comparison groups?', 'control group': r'control groups?',
               'quasi-experimental': r'quasi[- ]experimental'}
PRETEST_TERMS = {'pretest': r'\bpre-?tests?\b', 'posttest': r'\bpost-?tests?\b',
                 'ANCOVA': r'\bANCOVA\b'}
RANDOM_TERMS = {'randomly assigned': r'(?<!not )randomly assigned',
                'random assignment': r'(?<!no )(?<!without )(?<!non)(?<!non-)random assignment',
                'randomized': r'(?<!non-)(?<!non)(?<!not )\brandomi[sz]ed\b',
                'randomization': r'(?<!no )(?<!without )\brandomi[sz]ation\b'}
QED_TERMS = {'quasi-experimental': r'quasi[- ]experimental', 'matched': r'\bmatched\b',
             'nonequivalent': r'non-?equivalent', 'intact classes': r'intact (?:classes|groups)',
             'nonrandom': r'\bnon-?random\w*', 'not randomized': r'not randomi[sz]ed',
             'not randomly assigned': r'not randomly assigned',
             'no random assignment': r'(?:no|without) random assignment'}
SCD_DESIGNS = {'multiple_baseline': r'multiple[- ]baseline', 'multiple_probe': r'multiple[- ]probe',
               'alternating': r'alternating[- ]treatments?',
               'changing_criterion': r'changing[- ]criterion',
               'reversal': r'reversal design|withdrawal design|\bA-?B-?A-?B\b'}
DECREASE_TERMS = {'problem behavior': r'problem behaviou?rs?', 'challenging behavior': r'challenging behaviou?rs?',
                  'disruptive': r'\bdisrupti\w*', 'aggression': r'\baggressi\w*',
                  'self-injury': r'self[- ]injur\w*', 'off-task': r'off[- ]task',
                  'tantrum': r'\btantrums?\b', 'stereotypy': r'\bstereotyp\w*',
                  'number of errors': r'number of errors'}
QUALITATIVE_TERMS = {'interview': r'\binterview\w*', 'theme': r'\bthem(?:e|es|atic)\b',
                     'focus group': r'\bfocus groups?\b', 'qualitative': r'\bqualitative\b'}
HEADING = r'^[ \t]*(?:chapter\s+\w+\s*)?(?:{})\b[^\n]{{0,40}}$'
METHOD_START = re.compile(HEADING.format(r'methods?|methodology|participants|experimental design'
                                         r'|research design|design|procedures?'), re.I | re.M)
METHOD_END = re.compile(HEADING.format('results|findings|discussion|references'), re.I | re.M)
REFERENCES = re.compile(HEADING.format('references'), re.I | re.M)
CORRELATION = re.compile('correlat', re.I)
SCD_CAPTION = dict(CAPTION_TYPES)['scd_line']


def method_text(pages):
    """Text from the first Method heading to Results, or all text before References."""
    text = '\n'.join(page.get('text') or '' for page in pages)
    start = METHOD_START.search(text)
    end = (METHOD_END if start else REFERENCES).search(text, start.end() if start else 0)
    return text[start.start() if start else 0:end.start() if end else len(text)]


def is_correlation_table(grid):
    """True when the caption or header row of a table grid names correlations."""
    return bool(CORRELATION.search(' '.join([grid.get('caption') or '',
                                             *(str(cell or '') for cell in (grid.get('rows') or [[]])[0])])))


def _found(terms, text):
    counts = {label: len(re.findall(pattern, text, re.I)) for label, pattern in terms.items()}
    return [f"{label} ({count})" for label, count in counts.items() if count], sum(counts.values())


CAPTION_LINE = re.compile(r'(?im)^[ \t]*(?:figure|fig\.)\s*(\d+)\s*[.:|][ \t]*(.*)$')


def figure_captions(pages):
    """Figure numbers and caption lines found in page text, for ``suggest_design``."""
    return [{'figure': int(m.group(1)), 'caption': m.group(0).strip()}
            for page in pages for m in CAPTION_LINE.finditer(page.get('text') or '')]


def suggest_assignment(pages):
    """'rct' or 'qed' for a group study, and the evidence, from its Method text.

    Random-assignment terms against quasi-experimental terms by count; a tie
    gives 'rct' with the tied terms, and no terms gives 'rct' with no
    evidence, each for a coder to check.
    """
    text = method_text(pages)
    (rct, rct_n), (qed, qed_n) = _found(RANDOM_TERMS, text), _found(QED_TERMS, text)
    return (('qed', qed) if qed_n > rct_n else ('rct', rct) if rct_n > qed_n
            else ('rct', ['tie: ' + ', '.join(rct + qed)] if rct_n else []))


def suggest_scd_design(pages):
    """The single-case design named most often in the Method text, and the evidence.

    Multiple baseline, multiple probe, alternating treatments, changing
    criterion, or reversal (reversal, withdrawal, ABAB); a tie goes to the
    first in that order, and no design term gives 'ab' with no evidence, for
    a coder to check.
    """
    counts = {design: len(re.findall(pattern, method_text(pages), re.I))
              for design, pattern in SCD_DESIGNS.items()}
    design = max(counts, key=counts.get)
    return (design, [f"{design.replace('_', ' ')} ({counts[design]})"]) if counts[design] else ('ab', [])


def suggest_goal(pages, caption=''):
    """'decrease' or 'increase' as the intervention goal of a single-case graph, and the evidence.

    Behavior to reduce (problem or challenging behavior, disruption,
    aggression, self-injury, off-task, tantrums, stereotypy, number of errors)
    in the caption or Method text suggests 'decrease'; otherwise 'increase'
    with no evidence, for a coder to check.
    """
    found = _found(DECREASE_TERMS, f"{caption}\n{method_text(pages)}")[0]
    return ('decrease', found) if found else ('increase', [])


def suggest_design(pages, grids=(), figures=()):
    """'group', 'single_case', 'correlational', or 'qualitative', and the evidence for it.

    Design terms in the Method text decide first, a tie going to single-case
    since single-case papers name control groups and random assignment; then a correlation table,
    a single-case figure caption, pretest or ANCOVA words, and interview or
    theme words, in that order. With nothing found, the suggestion is 'group'.
    """
    text = method_text(pages)
    (scd, scd_n), (group, group_n) = _found(SINGLE_CASE_TERMS, text), _found(GROUP_TERMS, text)
    tables = [f"correlation table {grid['table']}" for grid in grids if is_correlation_table(grid)]
    captions = [f"figure {item['figure']} caption" for item in figures
                if re.search(SCD_CAPTION, item.get('caption') or '', re.I)]
    pretest, qualitative = _found(PRETEST_TERMS, text)[0], _found(QUALITATIVE_TERMS, text)[0]
    steps = [('single_case', scd if scd_n >= group_n else []),
             ('group', group if group_n > scd_n else []),
             ('correlational', tables), ('single_case', captions), ('group', pretest),
             ('qualitative', qualitative if len(qualitative) > 1 else []),
             ('group', ['no design terms found'])]
    return next((design, evidence) for design, evidence in steps if evidence)
