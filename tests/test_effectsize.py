import csv

import pytest

from pubextractor.comparisons import build_comparisons, text_statistics
from pubextractor.effectsize import (
    EFFECT_COLUMNS,
    LIMITS,
    METAFOR_COLUMNS,
    METHODS,
    allocate_n,
    cluster_terms,
    design_effect,
    effect_rows,
    effect_sizes,
    hedges_j,
    pooled_sd,
    sd_from_mse,
    sd_from_se,
    to_metafor,
)
from pubextractor.output import write_csv
from pubextractor.tables import extract_table_values, table_grid

approx = pytest.approx


class TestFormulas:
    def test_hedges_j(self):
        assert hedges_j(78) == approx(0.9903537, abs=1e-7)
        assert hedges_j(59.4432990) == approx(0.9873296, abs=1e-7)

    def test_pooled_sd(self):
        assert pooled_sd(12.32, 11.57, 39, 41) == approx(11.9412704, abs=1e-7)

    def test_design_effect(self):
        assert design_effect(80, 8, 0.20) == approx(2.8)

    def test_cluster_terms(self):
        gamma, df = cluster_terms(80, 8, 0.20)
        assert gamma == approx(0.9538462, abs=1e-7) and df == approx(59.4432990, abs=1e-7)

    @pytest.mark.parametrize('total, weights, expected', [
        (45, [1, 1], [22, 22]), (47, [1, 1], [24, 24]), (60, [2, 1], [40, 20])])
    def test_allocate_n_rounds_half_to_even(self, total, weights, expected):
        assert allocate_n(total, weights) == expected

    def test_sd_from_se(self):
        assert sd_from_se(2.0, 25) == approx(10.0)
        assert sd_from_se(1.5, 36, r2=0.49) == approx(12.6025208, abs=1e-7)

    def test_sd_from_mse(self):
        assert sd_from_mse(51.0, 0.49) == approx(10.0)


def rec(arm, group, stat, value, time='post'):
    return {'paper_id': 'S01', 'table': 1, 'outcome': 'Reading', 'time': time, 'arm': arm,
            'group': group, 'stat': stat, 'value': value, 'status': 'extracted',
            'quote': f'{arm} | {stat}: {value}'}


def arms(treatment, control, time='post'):
    return ([rec('Tutoring', 'treatment', stat, value, time) for stat, value in treatment.items()]
            + [rec('Usual', 'control', stat, value, time) for stat, value in control.items()])


def effect(records, methods=METHODS, allow_did=False, inputs=None, **study):
    found = build_comparisons(records, studies={'S01': study}, inputs=inputs)
    return effect_sizes(found, methods, allow_did)[0]


def values(found, *keys):
    return tuple(found[key] for key in keys)


WWC = arms({'mean': 59.12, 'sd': 12.32, 'n': 39}, {'mean': 54.34, 'sd': 11.57, 'n': 41})
NS = arms({'n': 39}, {'n': 41})
DID = (arms({'mean': 58.0, 'sd': 10.0, 'n': 20}, {'mean': 54.0, 'sd': 10.0, 'n': 20})
       + arms({'mean': 48.0}, {'mean': 50.0}, time='pre'))


class TestMethods:
    def test_posttest_difference_matches_wwc_table_e1(self):
        found = effect(WWC, design='rct')
        assert values(found, 'method', 'status', 'imputed_route', 'assumptions', 'df') == (
            'posttest_difference', 'computed', None, {}, 78)
        assert found['d'] == approx(0.4002924, abs=1e-7)
        assert found['j'] == approx(0.9903537, abs=1e-7)
        assert found['g'] == approx(0.3964311, abs=1e-7)
        assert found['var_d'] == approx(0.0510327, abs=1e-7)
        assert found['var_g'] == approx(0.0500341, abs=1e-7)
        assert found['se_g'] == approx(0.2236830, abs=1e-7)

    def test_from_t(self):
        found = effect(NS + text_statistics('t(78) = 1.79', paper_id='S01'), design='rct')
        assert found['method'] == 'from_t' and found['g'] == approx(0.3965191, abs=1e-7)

    def test_from_f_takes_its_sign_from_the_means(self):
        stat = text_statistics('F(1, 78) = 3.20', paper_id='S01')
        up = effect(arms({'mean': 59.12, 'n': 39}, {'mean': 54.34, 'n': 41}) + stat, design='rct')
        down = effect(arms({'mean': 54.34, 'n': 39}, {'mean': 59.12, 'n': 41}) + stat, design='rct')
        assert up['method'] == 'from_f' and up['g'] == approx(0.3962653, abs=1e-7)
        assert down['g'] == approx(-0.3962653, abs=1e-7)

    def test_f_with_more_than_one_numerator_df_is_not_used(self):
        found = effect(NS + text_statistics('F(2, 77) = 3.20', paper_id='S01'), design='rct')
        assert values(found, 'method', 'g', 'status') == (None, None, 'not_computed')
        assert found['checks'] == ['no_method', 'f_not_used']

    def test_difference_in_differences_is_off_by_default(self):
        found = effect(DID, design='qed')
        assert values(found, 'method', 'checks') == ('posttest_difference', ['qed_unadjusted'])

    def test_difference_in_differences(self):
        found = effect(DID, allow_did=True, design='qed')
        assert found['method'] == 'difference_in_differences'
        assert found['d'] == approx(0.6) and found['g'] == approx(0.5880795, abs=1e-7)
        assert found['var_d'] == approx(0.0645) and found['var_g'] == approx(0.0617925, abs=1e-7)
        assert found['assumptions'] == {'rho': {'value': 0.70, 'source': 'default'}}
        assert found['limits'] == [LIMITS['did']] and found['status'] == 'computed'

    def test_difference_in_differences_with_study_rho(self):
        found = effect(DID, allow_did=True, design='qed', rho=0.5)
        assert found['var_d'] == approx(0.1045)
        assert found['assumptions']['rho'] == {'value': 0.5, 'source': 'study'}

    def test_adjusted_difference_with_group_mean_ses(self):
        found = effect(arms({'adj_mean': 58.53, 'sd': 12.32, 'se': 1.5, 'n': 39},
                            {'adj_mean': 54.94, 'sd': 11.57, 'se': 1.4, 'n': 41}), design='rct')
        assert found['method'] == 'adjusted_difference'
        assert found['g'] == approx(0.2977380, abs=1e-7)
        assert found['var_g'] == approx(0.0295010, abs=1e-7)


class TestImputation:
    def test_sd_from_ancova_mse(self):
        records = (arms({'adj_mean': 55.0, 'n': 25}, {'adj_mean': 51.0, 'n': 25})
                   + text_statistics('MSE = 51.00', paper_id='S01'))
        found = effect(records, design='qed', rho=0.70)
        assert values(found, 'method', 'imputed_route') == ('adjusted_difference', 'sd_from_mse')
        assert found['g'] == approx(0.3937173, abs=1e-7)
        assert found['var_g'] == approx(0.0790085, abs=1e-7)
        assert found['assumptions'] == {'rho': {'value': 0.70, 'source': 'study'},
                                        'r2_sd': {'value': approx(0.49), 'source': 'default'},
                                        'r2_var': {'value': 0.0, 'source': 'default'}}
        assert values(found, 'status', 'checks') == ('review', ['r2_imputed'])

    def test_sd_from_ancova_mse_with_study_r2(self):
        records = (arms({'adj_mean': 55.0, 'n': 25}, {'adj_mean': 51.0, 'n': 25})
                   + text_statistics('MSE = 51.00', paper_id='S01'))
        found = effect(records, design='qed', r2=0.36)
        assert found['d'] == approx(0.4480896, abs=1e-7)
        assert found['g'] == approx(0.4410516, abs=1e-7)
        assert found['var_g'] == approx(0.0514889, abs=1e-7)
        assert found['assumptions'] == {'r2_sd': {'value': 0.36, 'source': 'study'},
                                        'r2_var': {'value': 0.36, 'source': 'study'}}
        assert values(found, 'status', 'checks') == ('computed', [])

    def test_sd_from_unadjusted_se(self):
        found = effect(arms({'mean': 55.0, 'se': 2.0, 'n': 25}, {'mean': 51.0, 'se': 2.0, 'n': 25}),
                       design='rct')
        assert values(found, 'method', 'imputed_route', 'assumptions') == (
            'posttest_difference', 'sd_from_se', {})
        assert found['d'] == approx(0.4)

    def test_sd_from_adjusted_se_uses_rho_squared(self):
        found = effect(arms({'adj_mean': 55.0, 'se': 1.4, 'n': 25},
                            {'adj_mean': 51.0, 'se': 1.4, 'n': 25}), design='rct')
        assert found['d'] == approx(0.4080816, abs=1e-7)
        assert found['g'] == approx(0.4016720, abs=1e-7)
        assert found['var_g'] == approx(0.0410915, abs=1e-7)
        assert found['assumptions'] == {'rho': {'value': 0.70, 'source': 'default'},
                                        'r2_sd': {'value': approx(0.49), 'source': 'default'}}
        assert values(found, 'status', 'checks') == ('review', ['r2_imputed'])

    def test_sd_from_adjusted_se_with_study_r2(self):
        found = effect(arms({'adj_mean': 55.0, 'se': 1.4, 'n': 25},
                            {'adj_mean': 51.0, 'se': 1.4, 'n': 25}), design='rct', r2=0.36)
        assert found['d'] == approx(0.4571429, abs=1e-7)
        assert found['g'] == approx(0.4499626, abs=1e-7)
        assert found['var_g'] == approx(0.0515658, abs=1e-7)
        assert found['assumptions'] == {'r2_sd': {'value': 0.36, 'source': 'study'}}
        assert values(found, 'status', 'checks') == ('computed', [])

    def test_allocated_n_is_recorded(self):
        found = effect([r for r in WWC if r['stat'] != 'n'], design='rct', n_total=80)
        assert values(found, 'n_t', 'n_c', 'imputed_route') == (40, 40, 'n_allocated')


class TestOrderAndChecks:
    def test_adjusted_means_come_first(self):
        found = effect(WWC + arms({'adj_mean': 58.53}, {'adj_mean': 54.94}), design='rct')
        assert values(found, 'method', 'limits') == ('adjusted_difference', [])
        assert found['g'] == approx(0.2977380, abs=1e-7)

    def test_other_order_notes_passed_over_adjusted_means(self):
        records = WWC + arms({'adj_mean': 58.53}, {'adj_mean': 54.94})
        found = effect(records, METHODS[1:] + METHODS[:1], design='rct')
        assert found['method'] == 'posttest_difference' and found['limits'] == [LIMITS['adjusted']]

    def test_qed_without_baseline_adjustment_needs_review(self):
        found = effect(WWC, design='qed')
        assert values(found, 'method', 'status', 'checks') == (
            'posttest_difference', 'review', ['qed_unadjusted'])

    def test_unknown_design_needs_review(self):
        assert effect(WWC)['checks'] == ['design_unknown']

    def test_t_sign_against_the_means_is_flagged(self):
        records = (arms({'mean': 54.34, 'n': 39}, {'mean': 59.12, 'n': 41})
                   + text_statistics('t(78) = 1.79', paper_id='S01'))
        found = effect(records, design='rct')
        assert found['g'] > 0 and found['checks'] == ['t_sign_differs_from_means']

    @pytest.mark.parametrize('change, check', [
        ({'sd': 0.0}, 'sd_not_positive'), ({'n': 1}, 'n_below_2')])
    def test_failed_checks_leave_g_empty(self, change, check):
        records = arms({'mean': 59.12, 'sd': 12.32, 'n': 39, **change},
                       {'mean': 54.34, 'sd': 11.57, 'n': 41})
        found = effect(records, design='rct')
        assert values(found, 'g', 'var_g', 'status') == (None, None, 'not_computed')
        assert check in found['checks']

    def test_rho_out_of_range_is_not_computed(self):
        found = effect(DID, allow_did=True, design='qed', rho=1.2)
        assert found['g'] is None and 'rho_out_of_range' in found['checks']

    def test_missing_control_is_not_computed(self):
        found = effect([r for r in WWC if r['arm'] == 'Tutoring'], design='rct')
        assert found['status'] == 'not_computed'
        assert found['checks'] == ['missing_control', 'n_missing']

    def test_check_only_comparison_passes_through(self):
        found = effect_sizes(build_comparisons(text_statistics('F(1, 40) = 4.10', paper_id='S02')))
        assert values(found[0], 'paper_id', 'g', 'status', 'checks') == (
            'S02', None, 'not_computed', ['test_statistic_unpaired'])

    def test_statistic_shared_by_two_outcomes_is_not_used(self):
        records = (WWC + [{**r, 'outcome': 'Math'} for r in WWC]
                   + text_statistics('t(78) = 1.79', paper_id='S01'))
        found = effect_sizes(build_comparisons(records, studies={'S01': {'design': 'rct'}}))
        assert [values(e, 'outcome', 'method', 'status', 'checks') for e in found] == [
            (outcome, 'posttest_difference', 'review', ['statistic_shared'])
            for outcome in ('Reading', 'Math')]
        assert found[1]['g'] == approx(0.3964311, abs=1e-7)


ADJUSTED_ROWS = [['Treatment', '39', '59.12', '12.32', '58.53', '1.5'],
                 ['Control', '41', '54.34', '11.57', '54.94', '1.4']]


def layout(rows, **study):
    records = extract_table_values([table_grid({'table': 1, 'rows': rows})], paper_id='S01')
    return effect_sizes(build_comparisons(records, studies={'S01': study}))


class TestTableLayouts:
    @pytest.mark.parametrize('header', [['Group', 'n', 'M', 'SD', 'Adjusted M', 'SE'],
                                        ['Group', 'n', 'Post M', 'Post SD', 'Adj. Post M', 'SE']])
    def test_adjusted_means_and_ses_beside_raw_values(self, header):
        found = layout([header] + ADJUSTED_ROWS, design='rct')
        assert [values(e, 'es_id', 'method', 'status') for e in found] == [
            ('S01:1:post:Treatment', 'adjusted_difference', 'computed')]
        assert found[0]['g'] == approx(0.2977380, abs=1e-7)
        assert found[0]['var_g'] == approx(0.0295010, abs=1e-7)

    def test_adjusted_means_and_ses_only(self):
        found = layout([['Group', 'n', 'Adj. M', 'SE']] + [r[:2] + r[4:] for r in ADJUSTED_ROWS],
                       design='rct')
        assert [values(e, 'method', 'imputed_route', 'status', 'checks') for e in found] == [
            ('adjusted_difference', 'sd_from_se', 'review', ['r2_imputed'])]
        assert found[0]['g'] == approx(0.2770977, abs=1e-7)
        assert found[0]['var_g'] == approx(0.0255525, abs=1e-7)

    def test_missing_sd_stops_and_names_the_arm(self):
        found = layout([['Group', 'n', 'M', 'SD'], ['Treatment', '20', '58.0', '—'],
                        ['Control', '21', '52.0', '9.0']], design='rct')
        assert values(found[0], 'g', 'status', 'checks') == (
            None, 'not_computed', ['sd_missing: Treatment (sd)', 'no_method'])


CLUSTER = {'design': 'rct', 'assignment': 'cluster', 'clusters': 8}


class TestClusterAssignment:
    def test_wwc_cluster_example(self):
        found = effect(WWC, **CLUSTER)
        assert found['g'] == approx(0.3859924, abs=1e-7)
        assert found['df'] == approx(59.4432990, abs=1e-7)
        assert found['j'] == approx(0.9873296, abs=1e-7)
        assert found['design_effect'] == approx(2.8)
        assert found['var_d'] == approx(0.1414353, abs=1e-7)
        assert found['var_g'] == approx(0.1377818, abs=1e-7)
        assert found['se_g'] == approx(0.3711897, abs=1e-7)
        assert found['assumptions'] == {'icc': {'value': 0.20, 'source': 'default'}}
        assert found['limits'] == [LIMITS['cluster']] and found['status'] == 'computed'

    def test_icc_default_for_behavioral_outcomes(self):
        found = effect(WWC, inputs={'S01:1:Reading:post:Tutoring': {'domain': 'behavioural'}},
                       **CLUSTER)
        assert found['domain'] == 'behavioural'
        assert found['assumptions']['icc'] == {'value': 0.10, 'source': 'default'}
        assert found['design_effect'] == approx(1.9)
        assert found['df'] == approx(72.5986497, abs=1e-7)
        assert found['g'] == approx(0.3915452, abs=1e-7)
        assert found['var_g'] == approx(0.0941328, abs=1e-7)

    def test_icc_from_study_overrides_the_domain(self):
        found = effect(WWC, inputs={'S01:1:Reading:post:Tutoring': {'domain': 'behavioural'}},
                       icc=0.05, **CLUSTER)
        assert found['assumptions']['icc'] == {'value': 0.05, 'source': 'study'}
        assert found['design_effect'] == approx(1.45)

    def test_t_is_taken_as_uncorrected_and_flagged(self):
        found = effect(NS + text_statistics('t(78) = 1.79', paper_id='S01'), **CLUSTER)
        assert found['g'] == approx(0.3860781, abs=1e-7)
        assert found['var_g'] == approx(0.1377823, abs=1e-7)
        assert found['checks'] == ['cluster_test_assumed_uncorrected']
        assert found['assumptions']['cluster_corrected'] == {'value': False, 'source': 'default'}

    def test_author_corrected_t_is_not_corrected_twice(self):
        found = effect(NS + text_statistics('t(78) = 1.79', paper_id='S01'),
                       cluster_corrected=True, **CLUSTER)
        assert found['d'] == approx(0.6699661, abs=1e-7)
        assert found['g'] == approx(0.6614774, abs=1e-7)
        assert found['var_g'] == approx(0.1401479, abs=1e-7)
        assert found['assumptions']['cluster_corrected'] == {'value': True, 'source': 'study'}
        assert values(found, 'status', 'checks') == ('computed', [])

    @pytest.mark.parametrize('corrected, g, var_g, checks', [
        (False, 0.3769989, 0.2169241, ['cluster_se_assumed_uncorrected']),
        (True, 0.6559160, 0.2205955, [])])
    def test_cluster_group_mean_ses(self, corrected, g, var_g, checks):
        records = arms({'mean': 55.0, 'se': 2.0, 'n': 25}, {'mean': 51.0, 'se': 2.0, 'n': 25})
        found = effect(records, design='rct', assignment='cluster', clusters=5,
                       cluster_corrected=corrected)
        assert found['g'] == approx(g, abs=1e-7) and found['var_g'] == approx(var_g, abs=1e-7)
        assert found['checks'] == checks
        assert found['assumptions']['cluster_corrected'] == {'value': corrected, 'source': 'study'}

    def test_difference_in_differences_has_no_cluster_variance(self):
        found = effect(DID, allow_did=True, design='qed', assignment='cluster', clusters=4)
        assert found['g'] is not None and found['var_g'] is None
        assert values(found, 'status', 'checks') == ('review', ['no_cluster_did_variance'])

    def test_variance_override_is_recorded(self):
        found = effect(DID, allow_did=True, inputs={'S01:1:Reading:post:Tutoring': {'var_g': 0.09}},
                       design='qed', assignment='cluster', clusters=4)
        assert values(found, 'var_g', 'se_g') == (0.09, approx(0.3))
        assert found['assumptions']['var_g'] == {'value': 0.09, 'source': 'override'}

    @pytest.mark.parametrize('clusters, check', [(None, 'clusters_missing'),
                                                 (80, 'clusters_exceed_n')])
    def test_cluster_counts_are_checked(self, clusters, check):
        found = effect(WWC, design='rct', assignment='cluster', clusters=clusters)
        assert found['g'] is None and check in found['checks']

    def test_mse_route_records_the_cluster_correction(self):
        records = (arms({'adj_mean': 55.0, 'n': 25}, {'adj_mean': 51.0, 'n': 25})
                   + text_statistics('MSE = 51.00', paper_id='S01'))
        found = effect(records, r2=0.36, design='rct', assignment='cluster', clusters=5)
        assert values(found, 'imputed_route', 'checks') == ('sd_from_mse', [])
        assert found['assumptions']['cluster_corrected'] == {'value': False, 'source': 'default'}

    def test_icc_out_of_range(self):
        found = effect(WWC, icc=1.0, **CLUSTER)
        assert found['g'] is None and 'icc_out_of_range' in found['checks']


THREE_ARMS = [['Group', 'n', 'Pre M', 'Post M', 'Post SD'],
              ['Treatment A', '20', '50.0', '58.0', '10.0'],
              ['Treatment B', '20', '50.5', '55.0', '11.0'],
              ['Control', '21', '50.2', '52.0', '9.0']]


class TestInputsKeys:
    def test_unknown_inputs_key_raises_naming_it(self):
        with pytest.raises(ValueError, match='S01:9:Nobody'):
            build_comparisons(WWC, inputs={'S01:9:Nobody': {'domain': 'behavioural'}})

    def test_known_inputs_key_is_accepted(self):
        found = build_comparisons(WWC, inputs={'S01:1:Reading:post:Tutoring':
                                               {'domain': 'behavioural'}})
        assert found[0]['inputs'] == {'domain': 'behavioural'}


class TestExports:
    def test_effect_rows_flatten_assumptions_and_checks(self):
        rows = effect_rows([effect(DID, allow_did=True, design='qed'), effect(WWC)])
        assert list(rows[0]) == list(EFFECT_COLUMNS)
        assert values(rows[0], 'rho', 'rho_source', 'icc', 'domain', 'checks', 'limits') == (
            0.70, 'default', None, 'achievement', '', LIMITS['did'])
        assert rows[1]['checks'] == 'design_unknown'

    def test_metafor_rows_keep_arms_and_shared_control(self):
        records = extract_table_values([table_grid({'table': 2, 'rows': THREE_ARMS})],
                                       paper_id='S01')
        found = effect_sizes(build_comparisons(records, studies={'S01': {'design': 'rct'}}))
        rows = to_metafor(found)
        assert list(rows[0]) == list(METAFOR_COLUMNS)
        assert [values(r, 'study', 'es_id', 'arm', 'control_id', 'n_t', 'n_c') for r in rows] == [
            ('S01', 'S01:2:post:Treatment A', 'Treatment A', 'S01:2:Control', 20, 21),
            ('S01', 'S01:2:post:Treatment B', 'Treatment B', 'S01:2:Control', 20, 21)]
        assert rows[0]['yi'] == approx(0.6193328, abs=1e-7)
        assert rows[0]['vi'] == approx(found[0]['var_g'])
        assert all(LIMITS['shared'] in e['limits'] for e in found)

    def test_override_and_missing_variance_in_exports(self):
        cluster = dict(allow_did=True, design='qed', assignment='cluster', clusters=4)
        empty = effect(DID, **cluster)
        override = effect(DID, inputs={'S01:1:Reading:post:Tutoring': {'var_g': 0.09}}, **cluster)
        assert to_metafor([empty]) == [] and to_metafor([override])[0]['vi'] == 0.09
        assert effect_rows([override])[0]['var_g_source'] == 'override'

    def test_metafor_skips_effects_without_variance(self):
        failed = effect(arms({'mean': 59.12, 'sd': 0.0, 'n': 39},
                             {'mean': 54.34, 'sd': 11.57, 'n': 41}), design='rct')
        assert to_metafor([failed, effect(WWC, design='rct')])[0]['yi'] == approx(0.3964311,
                                                                                abs=1e-7)
        assert len(to_metafor([failed])) == 0

    def test_metafor_warns_with_count_and_ids_of_dropped_effects(self, caplog):
        cluster = dict(allow_did=True, design='qed', assignment='cluster', clusters=4)
        empty = effect(DID, **cluster)
        with caplog.at_level('WARNING', logger='pubextractor.effectsize'):
            to_metafor([empty, effect(WWC, design='rct')])
        assert 'dropped 1 effect without var_g' in caplog.text
        assert empty['es_id'] in caplog.text

    def test_metafor_is_quiet_when_nothing_is_dropped(self, caplog):
        with caplog.at_level('WARNING', logger='pubextractor.effectsize'):
            to_metafor([effect(WWC, design='rct')])
        assert caplog.text == ''

    def test_csv_round_trip(self, tmp_path):
        path = write_csv(to_metafor([effect(WWC, design='rct')]), tmp_path / 'effects.csv')
        with open(path, encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0]) == list(METAFOR_COLUMNS)
        assert rows[0]['es_id'] == 'S01:1:Reading:post:Tutoring'


def test_package_exports_plan_4_names():
    import pubextractor

    for name in ('text_statistics', 'build_comparisons', 'METHODS', 'compute_effect',
                 'effect_sizes', 'EFFECT_COLUMNS', 'METAFOR_COLUMNS', 'effect_rows', 'to_metafor',
                 'hedges_j', 'pooled_sd', 'design_effect', 'cluster_terms', 'allocate_n',
                 'sd_from_se', 'sd_from_mse'):
        assert name in pubextractor.__all__ and hasattr(pubextractor, name)


class TestClusterCounts:
    @pytest.mark.parametrize('clusters, check', [(1, 'clusters_too_few'),
                                                 ('x', 'clusters_invalid'),
                                                 (2.5, 'clusters_invalid')])
    def test_unusable_cluster_count_is_checked_not_raised(self, clusters, check):
        found = effect(WWC, design='rct', assignment='cluster', clusters=clusters)
        assert values(found, 'g', 'status') == (None, 'not_computed') and check in found['checks']

    def test_bad_cluster_count_leaves_other_papers_computed(self):
        other = [{**r, 'paper_id': 'S02'} for r in WWC]
        found = effect_sizes(build_comparisons(WWC + other, studies={
            'S01': {'assignment': 'cluster', 'clusters': 'x'},
            'S02': {'assignment': 'cluster', 'clusters': 8}}))
        assert [(c['paper_id'], c['status']) for c in found] == [
            ('S01', 'not_computed'), ('S02', 'review')]

    def test_invalid_cluster_count_does_not_raise_on_the_se_route(self):
        records = arms({'mean': 59.0, 'se': 2.0, 'n': 39}, {'mean': 54.0, 'se': 2.0, 'n': 41})
        found = effect(records, design='rct', assignment='cluster', clusters='x',
                       cluster_corrected=True)
        assert 'clusters_invalid' in found['checks']
