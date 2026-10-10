"""Hedges' g and its sampling variance for group-design comparisons."""

import logging
from math import copysign, isfinite, sqrt
from numbers import Real

logger = logging.getLogger(__name__)

RHO = 0.70
ICC = {'achievement': 0.20, 'behavioural': 0.10, 'other': 0.10}


def _finite_number(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
    except OverflowError:
        return False


def hedges_j(df):
    """Small-sample correction J = 1 - 3 / (4 df - 1)."""
    return 1 - 3 / (4 * df - 1)


def pooled_sd(sd_t, sd_c, n_t, n_c):
    return sqrt(((n_t - 1) * sd_t ** 2 + (n_c - 1) * sd_c ** 2) / (n_t + n_c - 2))


def design_effect(n_total, clusters, icc):
    """1 + (m - 1) ICC with m = analyzed N / clusters."""
    return 1 + (n_total / clusters - 1) * icc


def cluster_terms(n_total, clusters, icc):
    """Small-number-of-clusters factor gamma and the degrees of freedom for J."""
    m = n_total / clusters
    gamma = 1 - 2 * (m - 1) * icc / (n_total - 2)
    df = ((n_total - 2) - 2 * (m - 1) * icc) ** 2 / (
        (n_total - 2) * (1 - icc) ** 2 + m * (n_total - 2 * m) * icc ** 2
        + 2 * (n_total - 2 * m) * icc * (1 - icc))
    return gamma, df


def allocate_n(n_total, weights):
    """Split a total N over arms in proportion to ``weights``, rounding half to even."""
    total = sum(weights)
    return [round(n_total * weight / total) for weight in weights]


def sd_from_se(se, n, r2=0.0):
    """Unadjusted SD from the SE of a group mean; ``r2`` > 0 for covariate-adjusted SEs."""
    return se * sqrt(n) / sqrt(1 - r2)


def sd_from_mse(mse, r2):
    """Unadjusted posttest SD from the ANCOVA error mean square; ``r2`` is rho^2 for one pretest."""
    return sqrt(mse / (1 - r2))


METHODS = ('adjusted_difference', 'posttest_difference', 'from_t', 'from_f',
           'difference_in_differences')

LIMITS = {
    'adjusted': 'adjusted means also reported; WWC prefers the covariate-adjusted difference',
    'did': 'difference in differences is not a WWC 5.0 reviewer-applied adjustment',
    'cluster': 'cluster correction assumes equal cluster sizes',
    'shared': 'control group shared with another arm; effects are dependent',
}


def _both(arms, time, stat):
    values = [arms[role][time].get(stat) for role in ('treatment', 'control')]
    return None if None in values else values


def _assume(study, name, default, assumptions, field=None):
    value = study.get(field or name)
    assumptions[name] = {'value': default if value is None else value,
                         'source': 'default' if value is None else 'study'}
    return assumptions[name]['value']


def _r2_sd(study, assumptions):
    """R^2 for an SD conversion: the study's, else rho^2 recorded as a default."""
    reported = study.get('r2')
    value = _assume(study, 'rho', RHO, assumptions) ** 2 if reported is None else reported
    assumptions['r2_sd'] = {'value': value, 'source': 'default' if reported is None else 'study'}
    return value


def _cluster_count(study):
    value = study.get('clusters')
    whole = isinstance(value, float) and value.is_integer()
    return int(value) if whole or (isinstance(value, int) and not isinstance(value, bool)) else None


def _se_scale(comparison, assumptions):
    """sqrt(gamma / eta) for cluster-corrected group-mean SEs (WWC E.33), else 1."""
    study = comparison['study']
    n = sum(_both(comparison['arms'], 'post', 'n'))
    icc = assumptions.get('icc', {}).get('value', 0)
    usable = (study.get('assignment') == 'cluster' and study.get('cluster_corrected')
              and 1 < (_cluster_count(study) or n) < n and 0 <= icc < 1)
    if not usable:
        return 1.0
    clusters = _cluster_count(study)
    gamma, _ = cluster_terms(n, clusters, icc)
    return sqrt(gamma / design_effect(n, clusters, icc))


def _sd_source(comparison):
    arms = comparison['arms']
    return next((source for source, found in (
        ('reported', _both(arms, 'post', 'sd')), ('mse', comparison['tests'].get('mse')),
        ('se', _both(arms, 'post', 'se'))) if found), None)


def _method(comparison, methods, allow_did):
    arms, tests = comparison['arms'], comparison['tests']
    design = comparison['study'].get('design')
    means, pres = _both(arms, 'post', 'mean'), _both(arms, 'pre', 'mean')
    signed = means or _both(arms, 'post', 'adj_mean')
    sd = _sd_source(comparison) is not None
    f = tests.get('f')
    available = {
        'adjusted_difference': bool(_both(arms, 'post', 'adj_mean') and sd),
        'posttest_difference': bool(means and sd and (design == 'rct' or not pres
                                                      or not allow_did)),
        'from_t': 't' in tests,
        'from_f': bool(f and f['df'][:1] == [1] and signed),
        'difference_in_differences': bool(allow_did and means and pres and sd
                                          and design != 'rct'),
    }
    return next((name for name in methods if available[name]), None)


def _post_sd(comparison, adjusted, assumptions, routes):
    """Unadjusted posttest SDs: reported, else from the ANCOVA MSE, else from group-mean SEs."""
    arms, study = comparison['arms'], comparison['study']
    source = _sd_source(comparison)
    routes.extend([f'sd_from_{source}'] if source in ('mse', 'se') else [])
    if source == 'reported':
        return _both(arms, 'post', 'sd')
    r2 = _r2_sd(study, assumptions) if source == 'mse' or adjusted else 0.0
    if not 0 <= r2 < 1:
        return None
    if source == 'mse':
        return [sd_from_mse(comparison['tests']['mse']['value'], r2)] * 2
    scale = _se_scale(comparison, assumptions)
    return [sd_from_se(se, n, r2) * scale for se, n in zip(_both(arms, 'post', 'se'),
                                                           _both(arms, 'post', 'n'))]


def _difference(method, comparison, assumptions, routes):
    """Standardized difference d and the variance core before the design effect."""
    arms, study, tests = comparison['arms'], comparison['study'], comparison['tests']
    n_t, n_c = _both(arms, 'post', 'n')
    a = (n_t + n_c) / (n_t * n_c)
    means, pres = _both(arms, 'post', 'mean'), _both(arms, 'pre', 'mean')
    adjusted = _both(arms, 'post', 'adj_mean')
    if method == 'from_t':
        return tests['t']['value'] * sqrt(a), a, None
    if method == 'from_f':
        signed = means or adjusted
        return copysign(sqrt(tests['f']['value'] * a), signed[0] - signed[1]), a, None
    sds = _post_sd(comparison, method == 'adjusted_difference', assumptions, routes)
    if sds is None or min(sds) <= 0:
        return None, None, sds
    sd = pooled_sd(*sds, n_t, n_c)
    if method == 'posttest_difference':
        return (means[0] - means[1]) / sd, a, sds
    if method == 'difference_in_differences':
        rho = _assume(study, 'rho', RHO, assumptions)
        return ((means[0] - means[1]) - (pres[0] - pres[1])) / sd, 2 * (1 - rho) * a, sds
    ses = _both(arms, 'post', 'se')
    if ses and study.get('assignment') != 'cluster':
        return (adjusted[0] - adjusted[1]) / sd, (ses[0] ** 2 + ses[1] ** 2) / sd ** 2, sds
    r2 = _assume(study, 'r2_var', 0.0, assumptions, field='r2')
    return (adjusted[0] - adjusted[1]) / sd, a * (1 - r2), sds


def _stops(comparison, method, sds, assumptions):
    study, arms = comparison['study'], comparison['arms']
    ns = [arms[role]['post'].get('n') for role in ('treatment', 'control')]
    stated = study.get('clusters')
    clusters = _cluster_count(study)
    clustered = study.get('assignment') == 'cluster'
    value = {name: entry['value'] for name, entry in assumptions.items()}
    return [name for name, failed in (
        ('no_method', method is None and comparison['control'] is not None),
        ('n_missing', None in ns),
        ('n_below_2', None not in ns and min(ns) < 2),
        ('sd_not_positive', sds is not None and min(sds) <= 0),
        ('rho_out_of_range', not -1 < value.get('rho', 0) < 1),
        ('r2_out_of_range', not all(0 <= value.get(name, 0) < 1
                                    for name in ('r2_sd', 'r2_var'))),
        ('icc_out_of_range', not 0 <= value.get('icc', 0) < 1),
        ('clusters_missing', clustered and not stated),
        ('clusters_invalid', clustered and bool(stated) and clusters is None),
        ('clusters_too_few', clustered and clusters is not None and clusters < 2),
        ('clusters_exceed_n', clustered and bool(clusters) and None not in ns
         and clusters >= sum(ns)),
    ) if failed]


def _flags(comparison, method, routes, assumptions):
    study, arms, tests = comparison['study'], comparison['arms'], comparison['tests']
    design = study.get('design')
    clustered = study.get('assignment') == 'cluster'
    uncorrected = clustered and not study.get('cluster_corrected')
    means = _both(arms, 'post', 'mean')
    t = tests.get('t')
    r2 = assumptions.get('r2_sd', {})
    return [name for name, on in (
        ('design_unknown', design is None),
        ('qed_unadjusted', design == 'qed' and method in ('posttest_difference', 'from_t',
                                                         'from_f')),
        ('r2_imputed', r2.get('source') == 'default' and bool(r2.get('value'))),
        ('t_sign_differs_from_means', method == 'from_t' and bool(means)
         and (means[0] - means[1]) * t['value'] < 0),
        ('f_not_used', 'f' in tests and method != 'from_f'),
        ('cluster_test_assumed_uncorrected', uncorrected and method in ('from_t', 'from_f')),
        ('cluster_se_assumed_uncorrected', uncorrected and 'sd_from_se' in routes),
        ('no_cluster_did_variance', clustered and method == 'difference_in_differences'),
    ) if on]


def _limits(comparison, method):
    study, arms = comparison['study'], comparison['arms']
    adjusted = bool(arms) and _both(arms, 'post', 'adj_mean')
    return [LIMITS[name] for name, on in (
        ('adjusted', bool(adjusted) and method not in (None, 'adjusted_difference')),
        ('did', method == 'difference_in_differences'),
        ('cluster', study.get('assignment') == 'cluster'),
        ('shared', comparison.get('shared_control'))) if on]


def _record(comparison, method, routes, assumptions, checks):
    study, arms = comparison['study'], comparison['arms']
    ns = ([arms[role]['post'].get('n') for role in ('treatment', 'control')] if arms
          else [None, None])
    return {'paper_id': comparison['paper_id'], 'es_id': comparison['es_id'],
            'table': comparison['table'], 'outcome': comparison['outcome'],
            'time': comparison['time'], 'treatment': comparison['treatment'],
            'control': comparison['control'], 'control_id': comparison['control_id'],
            'shared_control': bool(comparison.get('shared_control')),
            'design': study.get('design'), 'assignment': study.get('assignment', 'individual'),
            'domain': comparison.get('inputs', {}).get('domain', 'achievement'),
            'method': method, 'imputed_route': '; '.join(routes) or None,
            'n_t': ns[0], 'n_c': ns[1], 'd': None, 'j': None, 'g': None, 'var_d': None,
            'var_g': None, 'se_g': None, 'df': None, 'design_effect': None,
            'assumptions': assumptions, 'checks': checks,
            'limits': _limits(comparison, method), 'status': 'not_computed',
            'quote': '; '.join(comparison['quotes'])}


def _scale(study, n, assumptions, method):
    """gamma, df for J, design effect, df of the g squared term, and the factor on d."""
    if study.get('assignment') != 'cluster':
        return 1.0, n - 2, 1.0, n, 1.0
    icc = assumptions['icc']['value']
    gamma, df = cluster_terms(n, _cluster_count(study), icc)
    eta = design_effect(n, _cluster_count(study), icc)
    corrected = study.get('cluster_corrected') and method in ('from_t', 'from_f')
    return (1.0, df, eta, df, sqrt(eta)) if corrected else (gamma, df, eta, df, 1.0)


def compute_effect(comparison, methods=METHODS, allow_did=False):
    """Hedges' g and its variance by the first method in ``methods`` the data allow.

    Difference in differences runs only with ``allow_did=True``. The record
    keeps the method, the imputation routes, the assumptions with their
    source (``default``, ``study``, or ``override``), the checks, and the
    stated limits. A failed check leaves ``g`` empty and the status
    ``not_computed``; any other check gives ``review``. The comparison's
    ``inputs`` may set ``domain`` (``achievement`` by default) and ``var_g``.
    """
    study, inputs = comparison['study'], comparison.get('inputs', {})
    clustered = study.get('assignment') == 'cluster'
    assumptions, routes = {}, list(comparison['routes'])
    if comparison['treatment'] is None:
        return _record(comparison, None, routes, assumptions, comparison['checks'])
    values = [(stat, value) for arm in comparison['arms'].values()
              for time in ('pre', 'post') for stat, value in arm[time].items()
              if value is not None]
    invalid = [f'{stat}_invalid' for stat, value in values
               if not _finite_number(value) or (stat == 'n' and not float(value).is_integer())]
    invalid += [f'{name}_invalid' for name in ('rho', 'r2', 'icc')
                if study.get(name) is not None and not _finite_number(study[name])]
    invalid += ['var_g_invalid'] if inputs.get('var_g') is not None and (
        not _finite_number(inputs['var_g']) or inputs['var_g'] < 0) else []
    invalid += ['statistic_implausible'] if any(
        not _finite_number(test.get('value'))
        or (stat in ('f', 'mse') and test['value'] < 0)
        or any(not _finite_number(df) or df <= 0 for df in test.get('df', []))
        for stat, test in comparison['tests'].items()) else []
    if invalid:
        return _record(comparison, None, routes, assumptions,
                       list(dict.fromkeys(comparison['checks'] + invalid)))
    if clustered:
        _assume(study, 'icc', ICC.get(inputs.get('domain', 'achievement'), ICC['other']),
                assumptions)
    method = _method(comparison, methods, allow_did) if comparison['control'] else None
    ns = _both(comparison['arms'], 'post', 'n') if comparison['control'] else None
    d, core, sds = (_difference(method, comparison, assumptions, routes)
                    if method and ns and min(ns) >= 2 else (None, None, None))
    if clustered and (method in ('from_t', 'from_f')
                      or {'sd_from_se', 'sd_from_mse'} & set(routes)):
        _assume(study, 'cluster_corrected', False, assumptions)
    stops = _stops(comparison, method, sds, assumptions)
    flags = _flags(comparison, method, routes, assumptions)
    record = _record(comparison, method, routes, assumptions,
                     list(dict.fromkeys(comparison['checks'] + stops + flags)))
    if stops or d is None:
        return record
    gamma, df, eta, term, lift = _scale(study, record['n_t'] + record['n_c'], assumptions,
                                        method)
    d = d * lift
    j = hedges_j(df)
    g = j * d * sqrt(gamma)
    var_d = None if 'no_cluster_did_variance' in flags else core * eta + d ** 2 / (2 * term)
    var_g = None if var_d is None else j ** 2 * (core * eta + g ** 2 / (2 * term))
    if inputs.get('var_g') is not None:
        var_g = inputs['var_g']
        assumptions['var_g'] = {'value': var_g, 'source': 'override'}
    record.update({'d': d, 'j': j, 'g': g, 'df': df, 'var_d': var_d, 'var_g': var_g,
                   'se_g': None if var_g is None else sqrt(var_g),
                   'design_effect': eta if clustered else None,
                   'status': 'review' if record['checks'] else 'computed'})
    return record


def effect_sizes(comparisons, methods=METHODS, allow_did=False):
    """``compute_effect`` for each comparison."""
    return [compute_effect(comparison, methods, allow_did) for comparison in comparisons]

EFFECT_COLUMNS = ('paper_id', 'es_id', 'table', 'outcome', 'time', 'treatment', 'control',
                  'control_id', 'shared_control', 'design', 'assignment', 'domain', 'method',
                  'imputed_route', 'n_t', 'n_c', 'd', 'j', 'g', 'var_d', 'var_g', 'se_g', 'df',
                  'design_effect', 'rho', 'rho_source', 'icc', 'icc_source', 'r2_sd',
                  'r2_sd_source', 'r2_var', 'r2_var_source', 'cluster_corrected',
                  'cluster_corrected_source', 'var_g_source', 'checks', 'limits', 'status',
                  'quote')

METAFOR_COLUMNS = ('study', 'es_id', 'yi', 'vi', 'arm', 'control_id', 'n_t', 'n_c', 'outcome',
                   'time', 'method', 'status')


def effect_rows(effects):
    """One flat row per effect, columns in ``EFFECT_COLUMNS`` order, for ``write_csv``."""
    flat = [{**effect, 'checks': '; '.join(effect['checks']),
             'limits': '; '.join(effect['limits']),
             'var_g_source': effect['assumptions'].get('var_g', {}).get('source'),
             **{f'{name}{suffix}': effect['assumptions'].get(name, {}).get(key)
                for name in ('rho', 'icc', 'r2_sd', 'r2_var', 'cluster_corrected')
                for suffix, key in (('', 'value'), ('_source', 'source'))}}
            for effect in effects]
    return [{column: row.get(column) for column in EFFECT_COLUMNS} for row in flat]


def to_metafor(effects):
    """Rows with ``yi`` and ``vi`` for metafor, one per effect with a variance.

    Effects without a variance are left out, and a warning names their ``es_id`` values.

    ``arm`` and ``control_id`` feed ``grp1`` and ``grp2``, and ``n_t`` and
    ``n_c`` feed ``w1`` and ``w2``, of ``metafor::vcalc``.
    """
    dropped = [e['es_id'] for e in effects if e['var_g'] is None]
    if dropped:
        logger.warning('to_metafor dropped %d effect%s without var_g: %s', len(dropped),
                       '' if len(dropped) == 1 else 's', ', '.join(map(str, dropped)))
    return [{'study': e['paper_id'], 'es_id': e['es_id'], 'yi': e['g'], 'vi': e['var_g'],
             'arm': e['treatment'], 'control_id': e['control_id'], 'n_t': e['n_t'],
             'n_c': e['n_c'], 'outcome': e['outcome'], 'time': e['time'],
             'method': e['method'], 'status': e['status']}
            for e in effects if e['var_g'] is not None]
