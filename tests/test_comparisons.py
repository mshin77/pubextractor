import pytest

from pubextractor.comparisons import (
    build_comparisons,
    suggest_roles,
    text_arm_n,
    text_statistics,
    text_total_n,
)
from pubextractor.effectsize import effect_sizes
from pubextractor.tables import extract_table_values, table_grid

ARMS = [['Group', 'n', 'Pre M', 'Pre SD', 'Post M', 'Post SD'],
        ['Treatment A', '20', '50.0', '10.0', '58.0', '10.0'],
        ['Treatment B', '20', '50.5', '9.0', '55.0', '11.0'],
        ['Control', '21', '50.2', '10.0', '52.0', '9.0']]


def table(rows, number=2):
    return extract_table_values([table_grid({'table': number, 'rows': rows})], paper_id='S01')


def rec(arm, group, stat, value, time='post', outcome='Reading', number=1, status='extracted'):
    return {'paper_id': 'S01', 'table': number, 'outcome': outcome, 'time': time, 'arm': arm,
            'group': group, 'stat': stat, 'value': value, 'status': status,
            'quote': f'{arm} | {stat}: {value}'}


def two_arms():
    return [rec('Tutoring', 'treatment', 'mean', 58.0), rec('Tutoring', 'treatment', 'sd', 10.0),
            rec('Tutoring', 'treatment', 'n', 20), rec('Usual', 'control', 'mean', 54.0),
            rec('Usual', 'control', 'sd', 10.0), rec('Usual', 'control', 'n', 20)]


def checks(comparisons):
    return [check for c in comparisons for check in c['checks']]


class TestTextStatistics:
    def test_t_f_and_mse_with_df_and_location(self):
        found = text_statistics('t(78) = 1.79, p = .077; F(1, 78) = 3.20, MSE = 51.00.',
                                paper_id='S01', outcome='Reading', time='post',
                                where={'pdf_page': 4, 'section': 'Results'})
        assert [(r['stat'], r['value'], r['df']) for r in found] == [
            ('t', 1.79, [78.0]), ('f', 3.2, [1.0, 78.0]), ('mse', 51.0, [])]
        assert (found[0]['quote'], found[0]['pdf_page'], found[0]['section'],
                found[0]['outcome'], found[0]['group']) == ('t(78) = 1.79', 4, 'Results',
                                                            'Reading', None)

    def test_no_statistics_gives_no_records(self):
        assert text_statistics('Scores rose.') == []


class TestPairing:
    def test_one_comparison_per_treatment_arm_with_shared_control(self):
        found = build_comparisons(table(ARMS))
        assert [(c['treatment'], c['control'], c['control_id'], c['shared_control'])
                for c in found] == [('Treatment A', 'Control', 'S01:2:Control', True),
                                    ('Treatment B', 'Control', 'S01:2:Control', True)]
        first = found[0]
        assert first['es_id'] == 'S01:2:post:Treatment A' and first['pairing'] == 'rule'
        assert first['arms']['treatment']['post'] == {'mean': 58.0, 'sd': 10.0, 'se': None,
                                                      'adj_mean': None, 'n': 20}
        assert first['arms']['control']['pre']['mean'] == 50.2
        assert first['arms']['control']['post']['n'] == 21 and first['checks'] == []

    def test_override_drops_or_names_the_control(self):
        found = build_comparisons(table(ARMS), pairing={('S01', 2): {'Treatment B': None,
                                                                     'Treatment A': 'Control'}})
        assert [(c['treatment'], c['pairing'], c['shared_control']) for c in found] == [
            ('Treatment A', 'human', False)]

    def test_override_naming_an_absent_control_is_flagged(self):
        found = build_comparisons(two_arms(), pairing={('S01', 1): {'Tutoring': 'Wait-list'}})
        assert found[0]['control'] == 'Wait-list'
        assert 'override_control_not_found' in found[0]['checks']

    def test_inputs_attach_by_es_id(self):
        found = build_comparisons(two_arms(), inputs={'S01:1:Reading:post:Tutoring': {
            'domain': 'behavioural'}})
        assert found[0]['inputs'] == {'domain': 'behavioural'}

    def test_several_controls_are_not_guessed(self):
        records = two_arms() + [rec('Wait-list', 'control', 'mean', 53.0)]
        found = build_comparisons(records)
        assert found[0]['control'] is None and found[0]['checks'] == ['several_controls']

    def test_missing_control_and_control_at_other_time(self):
        lone = [rec('Tutoring', 'treatment', 'mean', 58.0)]
        assert checks(build_comparisons(lone)) == ['missing_control']
        later = lone + [rec('Usual', 'control', 'mean', 54.0, time='follow_up')]
        assert checks(build_comparisons(later)) == ['control_at_other_time']

    def test_pretest_from_another_table_when_unique(self):
        records = two_arms() + [rec('Tutoring', 'treatment', 'mean', 48.0, time='pre', number=1),
                                rec('Usual', 'control', 'mean', 50.0, time='pre', number=3)]
        found = build_comparisons(records)[0]['arms']
        assert (found['treatment']['pre']['mean'], found['control']['pre']['mean']) == (48.0, 50.0)

    def test_duplicate_values_are_flagged_not_chosen(self):
        records = two_arms() + [rec('Tutoring', 'treatment', 'mean', 57.0)]
        found = build_comparisons(records)[0]
        assert found['arms']['treatment']['post']['mean'] is None
        assert 'treatment_post_mean_duplicate' in found['checks']


class TestStatisticsAndN:
    def test_text_t_attaches_to_its_comparison(self):
        records = two_arms() + text_statistics('t(38) = 1.26', paper_id='S01')
        assert build_comparisons(records)[0]['tests'] == {'t': {'value': 1.26, 'df': [38.0]}}

    def test_statistic_matching_two_arms_is_not_attached(self):
        found = build_comparisons(table(ARMS) + text_statistics('t(39) = 2.10', paper_id='S01'))
        assert [(c['tests'], c['checks']) for c in found] == [({}, ['statistic_shared'])] * 2

    def test_statistic_without_outcome_matching_two_outcomes_is_not_attached(self):
        records = (two_arms() + [{**r, 'outcome': 'Math'} for r in two_arms()]
                   + text_statistics('t(38) = 1.26', paper_id='S01'))
        found = build_comparisons(records)
        assert [(c['outcome'], c['tests'], c['checks']) for c in found] == [
            ('Reading', {}, ['statistic_shared']), ('Math', {}, ['statistic_shared'])]

    def test_two_t_values_are_ambiguous(self):
        records = two_arms() + text_statistics('t(38) = 1.26 and t(38) = 2.10', paper_id='S01')
        found = build_comparisons(records)[0]
        assert found['tests'] == {} and 't_ambiguous' in found['checks']

    def test_unpaired_statistic_is_reported(self):
        found = build_comparisons(text_statistics('F(1, 40) = 4.10', paper_id='S02'))
        assert [(c['paper_id'], c['checks']) for c in found] == [
            ('S02', ['test_statistic_unpaired'])]

    def test_total_n_allocated_half_to_even(self):
        records = [r for r in two_arms() if r['stat'] != 'n']
        found = build_comparisons(records, studies={'S01': {'n_total': 45}})[0]
        assert (found['arms']['treatment']['post']['n'], found['arms']['control']['post']['n']) == (
            22, 22)
        assert found['routes'] == ['n_allocated'] and found['checks'] == ['n_allocation_rounded']

    def test_allocation_weights(self):
        records = [r for r in two_arms() if r['stat'] != 'n']
        found = build_comparisons(records, studies={'S01': {'n_total': 60, 'allocation': {
            'Tutoring': 2, 'Usual': 1}}})[0]
        assert (found['arms']['treatment']['post']['n'], found['arms']['control']['post']['n']) == (
            40, 20)
        assert found['checks'] == []

    def test_no_allocation_when_another_arm_states_n(self):
        records = [r for r in two_arms() if (r['arm'], r['stat']) != ('Usual', 'n')]
        found = build_comparisons(records, studies={'S01': {'n_total': 45}})[0]
        assert found['arms']['control']['post']['n'] is None and found['checks'] == ['n_missing']


class TestFlags:
    def test_unsettled_arm_role_is_reported(self):
        records = two_arms() + [rec('Peer group', None, 'mean', 51.0)]
        assert checks(build_comparisons(records))[-1] == 'arm_role_unsettled'

    def test_values_needing_review_are_not_used_but_flagged(self):
        records = two_arms() + [rec('Tutoring', 'treatment', 'sd', None, status='review')]
        found = build_comparisons(records)[0]
        assert found['arms']['treatment']['post']['sd'] == 10.0
        assert found['checks'] == ['unconfirmed_values']

    def test_review_values_are_used_and_flag_the_comparison(self):
        records = [{**r, 'status': 'review'} if (r['arm'], r['stat']) == ('Tutoring', 'mean')
                   else r for r in two_arms()]
        found = build_comparisons(records)[0]
        assert found['arms']['treatment']['post']['mean'] == 58.0
        assert found['checks'] == ['values_review']
        assert effect_sizes([found])[0]['status'] == 'review'

    def test_review_value_in_another_table_leaves_the_comparison_clean(self):
        records = two_arms() + [rec('Tutoring', 'treatment', 'mean', 1.0, number=2,
                                    status='review')]
        assert [c['checks'] for c in build_comparisons(records) if c['table'] == 1] == [[]]

    def test_not_found_and_refused_values_are_never_used(self):
        records = [r for r in two_arms() if (r['arm'], r['stat']) != ('Tutoring', 'sd')] + [
            rec('Tutoring', 'treatment', 'sd', 9.0, status='refused')]
        assert build_comparisons(records)[0]['arms']['treatment']['post']['sd'] is None

    def test_missing_sd_names_the_arm(self):
        found = build_comparisons(table([['Group', 'n', 'M', 'SD'], ['Treatment', '20', '58.0', '—'],
                                         ['Control', '21', '52.0', '9.0']]))
        assert found[0]['checks'] == ['sd_missing: Treatment (sd)']

    def test_each_unsettled_arm_has_its_own_es_id(self):
        found = build_comparisons(table([['Group', 'n', 'M', 'SD'],
                                         ['Peer tutoring', '20', '58.0', '10.0'],
                                         ['Cross-age tutoring', '20', '55.0', '11.0'],
                                         ['Control', '21', '52.0', '9.0']]))
        assert [(c['es_id'], c['treatment'], c['checks']) for c in found] == [
            ('S01:2:Peer tutoring', None, ['arm_role_unsettled']),
            ('S01:2:Cross-age tutoring', None, ['arm_role_unsettled'])]
        assert found[0]['quotes'][0] == 'Peer tutoring | n: 20'

    def test_paper_without_comparisons_is_reported(self):
        records = [rec('Tutoring', 'treatment', 'mean', None, status='not_found'),
                   rec('Tutoring', None, 'sd', None, status='not_found')]
        assert [(c['paper_id'], c['es_id'], c['checks']) for c in build_comparisons(records)] == [
            ('S01', None, ['no_comparisons'])]

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match='at least one record'):
            build_comparisons([])


ADJUSTED = [['Treatment', '39', '59.12', '12.32', '58.53', '1.5'],
            ['Control', '41', '54.34', '11.57', '54.94', '1.4']]


class TestTimePoints:
    @pytest.mark.parametrize('header', [['Group', 'n', 'M', 'SD', 'Adjusted M', 'SE'],
                                        ['Group', 'n', 'Post M', 'Post SD', 'Adjusted M', 'SE']])
    def test_untimed_values_join_the_posttest_of_their_table(self, header):
        found = build_comparisons(table([header] + ADJUSTED))
        assert [(c['es_id'], c['checks']) for c in found] == [('S01:2:post:Treatment', [])]
        assert found[0]['arms']['treatment']['post'] == {'mean': 59.12, 'sd': 12.32, 'se': 1.5,
                                                         'adj_mean': 58.53, 'n': 39}

    def test_adjusted_only_table_gives_one_comparison(self):
        rows = [['Group', 'n', 'Adj. M', 'SE']] + [row[:2] + row[4:] for row in ADJUSTED]
        found = build_comparisons(table(rows))
        assert [(c['es_id'], c['checks']) for c in found] == [('S01:2:post:Treatment', [])]
        assert found[0]['arms']['control']['post'] == {'mean': None, 'sd': None, 'se': 1.4,
                                                       'adj_mean': 54.94, 'n': 41}

    def test_untimed_values_beside_a_pretest_are_flagged(self):
        found = build_comparisons(table([['Group', 'n', 'Pre M', 'M', 'SD'],
                                         ['Treatment', '20', '50.0', '58.0', '10.0'],
                                         ['Control', '21', '50.2', '52.0', '9.0']]))
        assert [(c['es_id'], c['time'], c['checks']) for c in found] == [
            ('S01:2:Treatment', None, ['time_unclear'])]


class TestImplausibleStatistics:
    @pytest.mark.parametrize('text', ['t(0) = 1.20', 'F(0, 38) = 1.20', 'F(1, 38) = -3.10'])
    def test_implausible_statistic_is_flagged_when_attached(self, text):
        found = build_comparisons(two_arms() + text_statistics(text, paper_id='S01'))
        assert 'statistic_implausible' in found[0]['checks']

    def test_plausible_statistic_is_not_flagged(self):
        found = build_comparisons(two_arms() + text_statistics('t(38) = -1.26', paper_id='S01'))
        assert 'statistic_implausible' not in found[0]['checks']


class TestPosttestN:
    def arms(self, post_n=True):
        pre = [rec('Tutoring', 'treatment', 'n', 15, time='pre'),
               rec('Usual', 'control', 'n', 14, time='pre')]
        post = [rec('Tutoring', 'treatment', 'n', 13), rec('Usual', 'control', 'n', 12)]
        return two_arms()[:2] + two_arms()[3:5] + pre + (post if post_n else [])

    def test_posttest_n_wins_over_pretest_n(self):
        found = effect_sizes(build_comparisons(self.arms()))[0]
        assert (found['n_t'], found['n_c'], found['imputed_route']) == (13, 12, None)

    def test_pretest_n_is_fallback_and_recorded(self):
        comparison = build_comparisons(self.arms(post_n=False))[0]
        found = effect_sizes([comparison])[0]
        assert (found['n_t'], found['n_c']) == (15, 14)
        assert 'n_from_pretest' in found['imputed_route']

    def test_n_from_another_table_of_the_paper_is_flagged(self):
        other = [rec('Tutoring', 'treatment', 'n', 18, number=3),
                 rec('Usual', 'control', 'n', 17, number=3),
                 rec('Usual', 'control', 'n', 99, number=3, outcome='Maths')]
        comparison = build_comparisons(self.arms(post_n=False)[:4] + other)[0]
        found = effect_sizes([comparison])[0]
        assert (found['n_t'], found['n_c'], found['status']) == (18, 17, 'review')
        assert 'n_from_other_source' in found['imputed_route']


ROLES = {'Tutoring': 'treatment', 'Usual': 'control'}


def test_suggested_role_is_flagged_until_confirmed():
    records = [{**r, 'group_source': 'suggested'} if r['group'] == 'treatment' else r
               for r in two_arms()]
    assert 'role_suggested' in checks(build_comparisons(records))
    assert 'role_suggested' not in checks(build_comparisons(two_arms()))


class TestSuggestRoles:
    def arm(self, label, group=None, table=2):
        return {'paper_id': 'S01', 'table': table, 'arm': label, 'group': group}

    def test_characteristics_and_statistic_rows_are_not_arms(self):
        found = suggest_roles([self.arm('Caucasian 137'), self.arm('SLD 24'), self.arm('7 92'),
                               self.arm('M 12.21'), self.arm('Total')])
        assert {role for role, _ in found.values()} == {'other'}
        assert found[('S01', 2, 'SLD 24')][1] == 'participant characteristic'
        assert found[('S01', 2, 'Total')][1] == 'statistic, total, or model row'

    @pytest.mark.parametrize('label', ['Grade-level instruction', 'Model-based instruction',
                                       'Mean-focused tutoring', 'Total program', 'MA tutoring',
                                       'Edmark (ED)', 'Range of motion', 'Visual supports'])
    def test_real_arm_labels_stay_open(self, label):
        assert suggest_roles([self.arm(label)])[('S01', 2, label)] == (None, '')

    @pytest.mark.parametrize('label', ['BA, BS 6', 'Grade', 'Years teaching', 'EBD 9', 'grade 4',
                                       '(n = 159)', 'Median 11.50', 'HLM Models'])
    def test_rows_that_are_not_arms(self, label):
        assert suggest_roles([self.arm(label)])[('S01', 2, label)][0] == 'other'

    def test_other_set_at_extraction_stays_other(self):
        found = suggest_roles([self.arm('Peers', 'other')])
        assert found == {('S01', 2, 'Peers'): ('other', 'set to other at extraction')}

    def test_distant_cue_and_peer_sentences_are_ignored(self):
        pages = {'S01': [{'page': 1, 'text': 'Students in the Edmark program read daily, while the '
                                              'control classroom used books. Edmark and Usual met.'}]}
        found = suggest_roles([self.arm('Edmark'), self.arm('Usual', 'control')], pages)
        assert found == {('S01', 2, 'Edmark'): ('treatment', 'other arm in the table is control')}

    def test_open_arm_beside_a_control_is_treatment(self):
        found = suggest_roles([self.arm('BAU', 'control'), self.arm('EAI')])
        assert found == {('S01', 2, 'EAI'): ('treatment', 'other arm in the table is control')}

    def test_two_open_arms_beside_a_control_stay_open(self):
        found = suggest_roles([self.arm('BAU', 'control'), self.arm('EAI'), self.arm('AITA')])
        assert {role for role, _ in found.values()} == {None}

    def test_text_cue_names_the_page(self):
        pages = {'S01': [{'page': 4, 'text': 'Students in EAI received the intervention daily.'}]}
        role, reason = suggest_roles([self.arm('EAI')], pages)[('S01', 2, 'EAI')]
        assert role == 'treatment' and reason.startswith('text p. 4: Students in EAI')

    def test_mixed_text_cues_give_no_suggestion(self):
        pages = {'S01': [{'page': 4, 'text': 'EAI was the intervention. EAI and the control met.'}]}
        assert suggest_roles([self.arm('EAI')], pages)[('S01', 2, 'EAI')] == (None, '')


class TestTextArmN:
    def values(self, text):
        return [(r['arm'], r['group'], r['value']) for r in text_arm_n(text, ROLES, 'S01')]

    def test_label_before_each_n(self):
        assert self.values('Pupils were assigned to tutoring (n = 24) or usual teaching '
                           '(n = 22).') == [('Tutoring', 'treatment', 24), ('Usual', 'control', 22)]

    def test_labels_in_a_row_pair_with_values_in_order(self):
        assert self.values('Students\nUsual Tutoring\n(n = 22) (n = 24) p') == [
            ('Usual', 'control', 22), ('Tutoring', 'treatment', 24)]

    def test_only_the_labels_next_to_the_values_pair(self):
        assert self.values('Tutoring and Usual classes met daily; Usual Tutoring (n = 22) '
                           '(n = 24)') == [('Usual', 'control', 22), ('Tutoring', 'treatment', 24)]

    def test_cluster_counts_are_not_arm_n(self):
        assert self.values('Teachers\nTutoring Usual\n(n = 6) (n = 5)') == []

    @pytest.mark.parametrize('text', ['Tutoring (n = 5 classes) and Usual (n = 5 classes)',
                                      'Of the Tutoring group, n = 5 withdrew',
                                      'Tutoring n = 2 dropped',
                                      'Tutoring (n = 24) withdrew'])
    def test_cluster_or_attrition_value_is_not_arm_n(self, text):
        assert self.values(text) == []

    def test_distant_label_does_not_pair(self):
        assert self.values('Tutoring ran all year in four sites. Reading Test (n = 46)') == []

    def test_record_keeps_the_quote_and_location(self):
        found = text_arm_n('Tutoring (n = 24)', ROLES, 'S01', where={'pdf_page': 3})[0]
        assert (found['stat'], found['table'], found['outcome'], found['pdf_page'],
                found['quote'], found['status']) == ('n', None, None, 3, 'Tutoring (n = 24)',
                                                     'extracted')

    @pytest.mark.parametrize('text, total', [('In all, N = 57 pupils took part.', 57),
                                             ('N = 57 at pretest and N = 51 at posttest.', None),
                                             ('Tutoring (n = 24)', None)])
    def test_total_n_is_one_stated_value(self, text, total):
        assert text_total_n(text) == total

    def test_comparison_takes_arm_n_from_text(self):
        records = [r for r in two_arms() if r['stat'] != 'n']
        text = text_arm_n('tutoring (n = 24) or usual teaching (n = 22)', ROLES, 'S01')
        found = effect_sizes(build_comparisons(records + text))
        assert len(found) == 1
        assert (found[0]['n_t'], found[0]['n_c'], found[0]['status']) == (24, 22, 'review')
        assert 'n_from_other_source' in found[0]['imputed_route']
