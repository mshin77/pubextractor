# tests/test_table_roles.py
import json

import pytest

from pubextractor.licensing import check_llm_use
from pubextractor.table_roles import find_groups, label_residue, map_headers
from pubextractor.tables import read_header, table_grid

LOCAL = dict(permission=check_llm_use(access='subscription', ai_terms='silent'),
             endpoint='http://localhost:11434/v1')


def responder(reply):
    sent = []

    def respond(prompt, images=None):
        sent.append(prompt)
        return reply if isinstance(reply, str) else json.dumps(reply)
    respond.sent = sent
    return respond


def header(*labels):
    rows = [['Measure', *labels], ['Reading', *['1'] * len(labels)]]
    return read_header(table_grid({'table': 1, 'rows': rows}))


def stat_of(label):
    return map_headers(header(label))[1]['stat']


class TestMapHeadersByRules:
    def test_common_headers(self):
        columns = map_headers(header('M', 'SD', 'Adj. M', 'SE', 'n', 'Pre M (SD)',
                                     'Posttest Mean', 'r', '%'))
        assert [c['stat'] for c in columns] == [None, 'mean', 'sd', 'adj_mean', 'se', 'n',
                                                'mean_sd', 'mean', 'r', 'percent']
        assert [c['time'] for c in columns][6:8] == ['pre', 'post']
        assert columns[0]['source'] == 'stub' and columns[1]['source'] == 'rule'

    @pytest.mark.parametrize('label, stat', [
        ('Mean', 'mean'), ('Std. Dev.', 'sd'), ('S.D.', 'sd'), ('S.E.', 'se'),
        ('Standard error', 'se'), ('Adjusted mean', 'adj_mean'), ('Adj. M (SE)', 'adj_mean_se'),
        ('M ± SD', 'mean_sd'), ('N', 'n'), ('Treatment M', 'mean'), ('Score', None),
    ])
    def test_variants(self, label, stat):
        assert stat_of(label) == stat

    @pytest.mark.parametrize('label, time', [
        ('Pretest M', 'pre'), ('Baseline SD', 'pre'), ('Post M', 'post'),
        ('Follow-up M', 'follow_up'), ('Delayed posttest M', 'follow_up'),
        ('Maintenance SD', 'follow_up'), ('M', None),
    ])
    def test_time_points(self, label, time):
        assert map_headers(header(label))[1]['time'] == time

    @pytest.mark.parametrize('label', ['Adj. M', 'Adj. M (SE)'])
    def test_adjusted_means_default_to_posttest(self, label):
        assert map_headers(header(label))[1]['time'] == 'post'

    def test_adjusted_mean_keeps_a_stated_time(self):
        assert map_headers(header('Follow-up adj. M'))[1]['time'] == 'follow_up'

    @pytest.mark.parametrize('label, time', [
        ('Adj. Post M', 'post'), ('Adjusted posttest mean', 'post'),
        ('Adj follow-up M', 'follow_up')])
    def test_adjusted_mean_with_a_time_word_between(self, label, time):
        column = map_headers(header(label))[1]
        assert (column['stat'], column['time']) == ('adj_mean', time)

    @pytest.mark.parametrize('label, residue', [
        ('Adj. Post M', ''), ('Adj. SE', ''), ('Adjusted SD', ''), ('Adj Reading M', 'Reading'),
        ('Reading adj. M (SE)', 'Reading')])
    def test_adjusted_words_leave_no_residue(self, label, residue):
        assert label_residue(label) == residue

    @pytest.mark.parametrize('label', ['95% CI', 'Mean difference', 'Mean change', 'Gain',
                                       "Cohen's d", "Hedges' g", 'ES', 'Effect size',
                                       'Confidence interval', 'Diff.'])
    def test_differences_and_effect_sizes_are_vetoed(self, label):
        column = map_headers(header(label))[1]
        assert (column['stat'], column['source']) == (None, 'veto')

    @pytest.mark.parametrize('label', ['Outcome', 'Measures', 'Variable', 'Group', 'Condition'])
    def test_label_columns_get_no_role(self, label):
        column = map_headers(header(label))[1]
        assert (column['stat'], column['source'], column['kind']) == (
            None, 'stub', 'outcome' if label[0] in 'OMV' else 'arm')

    def test_unmatched_without_model_stays_unmapped(self):
        column = map_headers(header('Score'))[1]
        assert (column['stat'], column['source']) == (None, 'none')


class TestMapHeadersWithModel:
    def test_model_maps_only_unmatched_headers_with_a_quote(self):
        respond = responder({'labels': [{'number': 1, 'choice': 'mean', 'quote': 'Score'}]})
        columns = map_headers(header('Score', 'SD'), respond=respond, **LOCAL)
        assert (columns[1]['stat'], columns[1]['source']) == ('mean', 'prompt')
        assert '[1] Score' in respond.sent[0] and '[2]' not in respond.sent[0]
        assert '"adj_mean"' in respond.sent[0]

    def test_vetoed_headers_are_not_sent(self):
        respond = responder({'labels': []})
        columns = map_headers(header('Mean difference', '95% CI'), respond=respond, **LOCAL)
        assert respond.sent == [] and [c['source'] for c in columns[1:]] == ['veto', 'veto']

    def test_long_header_quoted_with_other_spacing_is_kept(self):
        label = 'Reading comprehension score'
        respond = responder({'labels': [{'number': 1, 'choice': 'mean',
                                         'quote': 'reading  comprehension score'}]})
        column = map_headers(header(label), respond=respond, **LOCAL)[1]
        assert (column['stat'], column['source']) == ('mean', 'prompt')

    @pytest.mark.parametrize('pick', [
        {'number': 1, 'choice': 'median', 'quote': 'Score'},
        {'number': 1, 'choice': 'mean', 'quote': 'Average score'},
        {'number': 2, 'choice': 'mean', 'quote': 'Score'},
    ])
    def test_unlisted_choice_or_wrong_quote_stays_unmapped(self, pick):
        column = map_headers(header('Score'), respond=responder({'labels': [pick]}), **LOCAL)[1]
        assert (column['stat'], column['source']) == (None, 'none')

    def test_unparseable_reply_stays_unmapped(self):
        column = map_headers(header('Score'), respond=responder('not json'), **LOCAL)[1]
        assert column['stat'] is None

    def test_license_refusal_sends_nothing(self):
        respond = responder({'labels': []})
        refused = check_llm_use(access='subscription', ai_terms='forbids')
        column = map_headers(header('Score'), respond=respond, permission=refused,
                             endpoint='http://localhost:11434/v1')[1]
        assert column['source'] == 'refused' and 'forbid' in column['quote']
        assert respond.sent == []

    def test_missing_permission_is_flagged_not_raised(self):
        respond = responder({'labels': []})
        column = map_headers(header('Score'), respond=respond)[1]
        assert column['source'] == 'refused' and 'permission is required' in column['quote']
        assert respond.sent == []


class TestMapHeadersOverrides:
    def test_human_override_wins_and_skips_the_model(self):
        respond = responder({'labels': [{'number': 1, 'choice': 'mean', 'quote': 'Score'}]})
        columns = map_headers(header('Score', 'SD'),
                              overrides={'Score': {'stat': 'n', 'time': 'post'}, 3: 'se'},
                              respond=respond, **LOCAL)
        assert [(c['stat'], c['time'], c['source']) for c in columns[1:]] == [
            ('n', 'post', 'human'), ('se', None, 'human')]
        assert respond.sent == []

    def test_override_reaches_a_vetoed_header(self):
        column = map_headers(header('Mean change'), overrides={2: 'mean'})[1]
        assert (column['stat'], column['source']) == ('mean', 'human')

    @pytest.mark.parametrize('overrides, message', [
        ({2: 'median'}, 'stat must be one of'),
        ({9: 'sd'}, 'names no column'),
        ({2: {'time': 'later'}}, 'time must be one of'),
    ])
    def test_bad_overrides_are_reported(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            map_headers(header('Score'), overrides=overrides)


def groups(rows, **kwargs):
    grid = table_grid({'table': 1, 'rows': rows})
    return find_groups(grid, read_header(grid), **kwargs)


def roles(entries):
    return {(e['axis'], e['index']): (e['label'], e['group'], e['source']) for e in entries}


COLUMN_ARMS = [['', 'Intervention (n = 24)', '', 'Business as usual (n = 22)', ''],
               ['Measure', 'M', 'SD', 'M', 'SD'],
               ['Reading', '61.3', '14.2', '55.0', '13.1']]
ROW_ARMS = [['Group', 'n', 'M', 'SD'],
            ['Experimental', '24', '61.3', '14.2'],
            ['Wait-list control', '22', '55.0', '13.1']]
NAMED_ARMS = [['Group', 'M'], ['Peer tutoring', '61.3'], ['Teacher-led', '55.0']]


class TestFindGroupsByRules:
    def test_arms_in_header_columns(self):
        found = roles(groups(COLUMN_ARMS))
        assert found[('col', 2)] == ('Intervention', 'treatment', 'rule')
        assert found[('col', 5)] == ('Business as usual', 'control', 'rule')
        assert not any(axis == 'row' for axis, _ in found)

    def test_single_header_row_arms_lose_stat_words(self):
        found = roles(groups([['Measure', 'Treatment M', 'Control M'], ['Reading', '1', '2']]))
        assert found[('col', 2)][:2] == ('Treatment', 'treatment')

    def test_n_stated_with_the_arm(self):
        entry = next(e for e in groups(COLUMN_ARMS) if e['index'] == 4)
        assert (entry['n'], entry['n_row']) == (22, 1)

    def test_arms_in_rows(self):
        found = roles(groups(ROW_ARMS))
        assert found[('row', 2)][1:] == ('treatment', 'rule')
        assert found[('row', 3)][1:] == ('control', 'rule')

    def test_arm_column_beside_an_outcome_column_is_filled_down(self):
        rows = [['Outcome', 'Group', 'M'], ['Reading', 'Treatment', '61.3'],
                ['', 'Control', '55.0'], ['Math', 'Treatment', '48.0'], ['', '', '47.0']]
        found = roles(groups(rows))
        assert found[('row', 5)] == ('Treatment', 'treatment', 'rule')

    def test_two_treatment_arms_keep_their_labels(self):
        rows = [['Group', 'M'], ['Treatment A', '61.3'], ['Treatment B', '58.0'],
                ['Control', '55.0']]
        found = roles(groups(rows))
        assert found[('row', 2)][:2] == ('Treatment A', 'treatment')
        assert found[('row', 3)][:2] == ('Treatment B', 'treatment')

    def test_vetoed_columns_are_not_arms(self):
        found = roles(groups([['Measure', 'Treatment', 'Mean difference'], ['Reading', '1', '2']]))
        assert ('col', 3) not in found

    @pytest.mark.parametrize('label, group', [
        ('Comparison', 'control'), ('BAU', 'control'), ('Typical practice', 'control'),
        ('Waitlist', 'control'), ('Treatment as usual', 'control'),
        ('Treatment group', 'treatment'), ('Peer tutoring', None),
        ('No intervention', 'control'), ('Non-intervention', 'control'), ('Controls', 'control'),
        ('Waiting list', 'control'), ('Untreated', 'control'), ('Usual care', 'control'),
        ('Traditional instruction', 'control'), ('Placebo', 'control'), ('Treated', 'treatment'),
    ])
    def test_variants(self, label, group):
        found = roles(groups([['Group', 'M'], [label, '1']]))
        assert found[('row', 2)][1] == group

    def test_named_arms_without_model_stay_unlabeled(self):
        found = roles(groups(NAMED_ARMS))
        assert found[('row', 2)] == ('Peer tutoring', None, 'none')


class TestFindGroupsWithModel:
    def test_model_labels_named_arms_with_quotes(self):
        respond = responder({'labels': [
            {'number': 1, 'choice': 'treatment', 'quote': 'Peer tutoring'},
            {'number': 2, 'choice': 'control', 'quote': 'Teacher-led'}]})
        found = roles(groups(NAMED_ARMS, respond=respond, **LOCAL))
        assert found[('row', 2)] == ('Peer tutoring', 'treatment', 'prompt')
        assert found[('row', 3)] == ('Teacher-led', 'control', 'prompt')
        assert '["treatment", "control", "other"]' in respond.sent[0]

    def test_model_is_not_asked_when_rules_find_both_roles(self):
        respond = responder({'labels': []})
        groups(ROW_ARMS, respond=respond, **LOCAL)
        assert respond.sent == []

    def test_stat_words_are_not_sent(self):
        respond = responder({'labels': []})
        groups([['Group', 'M', 'SD'], ['Peer tutoring', '1', '2']], respond=respond, **LOCAL)
        assert '[1] Peer tutoring' in respond.sent[0] and '[2]' not in respond.sent[0]


class TestFindGroupsOverrides:
    def test_human_override_by_label_or_position(self):
        found = roles(groups(NAMED_ARMS, overrides={'Peer tutoring': 'treatment',
                                                    ('row', 3): 'other'}))
        assert found[('row', 2)] == ('Peer tutoring', 'treatment', 'human')
        assert found[('row', 3)] == ('Teacher-led', 'other', 'human')

    def test_override_must_name_a_role(self):
        with pytest.raises(ValueError, match='group must be one of'):
            groups(NAMED_ARMS, overrides={'Peer tutoring': 'active'})
