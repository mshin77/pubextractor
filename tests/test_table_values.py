# tests/test_table_values.py
import json

import pytest

from pubextractor.licensing import check_llm_use
from pubextractor.output import COLUMNS, to_long, to_wide
from pubextractor.tables import extract_table_values, read_tables, table_grid

LOCAL = dict(permission=check_llm_use(access='subscription', ai_terms='silent'),
             endpoint='http://localhost:11434/v1')


def values(rows, **kwargs):
    return extract_table_values([table_grid({'table': 1, 'rows': rows})], paper_id='S01', **kwargs)


def by_field(records):
    return {(r['field'], r['row'], r['col']): (r['value'], r['status']) for r in records}


def one(records, field):
    return next(r for r in records if r['field'] == field)


def identity(record):
    return (record['outcome'], record['arm'], record['group'], record['time'], record['stat'])


class TestTableValues:
    def test_docx_groups_in_columns(self, study_docx):
        records = extract_table_values(read_tables(study_docx), paper_id='S01')
        found = by_field(records)
        assert found[('treatment_mean', 3, 2)] == (61.3, 'extracted')
        assert found[('control_sd', 3, 5)] == (13.1, 'extracted')
        assert found[('treatment_n', 1, 2)] == (24, 'extracted')
        assert found[('control_n', 1, 4)] == (22, 'extracted')
        record = one(records, 'treatment_mean')
        assert (record['layer'], record['table'], record['section'], record['pdf_page']) == (
            'table', 1, 'Results', None)
        assert identity(record) == ('Reading', 'Treatment', 'treatment', None, 'mean')

    def test_groups_in_rows(self):
        records = values([['Group', 'n', 'M', 'SD'],
                          ['Treatment', '24', '61.3', '14.2'],
                          ['Control', '22', '55.0', '13.1']])
        found = by_field(records)
        assert found[('treatment_n', 2, 2)] == (24, 'extracted')
        assert found[('control_mean', 3, 3)] == (55.0, 'extracted')
        assert isinstance(found[('treatment_n', 2, 2)][0], int)
        assert one(records, 'control_mean')['outcome'] is None

    def test_combined_cells_and_dashes(self):
        records = values([['Group', 'Post M (SD)'],
                          ['Treatment (n = 24)', '61.3a (14.2)'],
                          ['Control (n = 22)', '—']])
        found = by_field(records)
        assert found[('treatment_post_mean', 2, 2)] == (61.3, 'extracted')
        assert found[('treatment_post_sd', 2, 2)] == (14.2, 'extracted')
        assert found[('control_post_mean_sd', 3, 2)] == (None, 'not_found')
        assert found[('treatment_n', 2, 1)] == (24, 'extracted')
        assert one(records, 'treatment_post_sd')['time'] == 'post'

    def test_not_reported_and_text_cells_are_flagged(self):
        found = by_field(values([['Group', 'M'], ['Treatment', 'n.r.'], ['Control', 'see text']]))
        assert found[('treatment_mean', 2, 2)] == (None, 'not_found')
        assert found[('control_mean', 3, 2)] == (None, 'review')

    def test_cell_that_does_not_fit_its_column_needs_review(self):
        found = by_field(values([['Group', 'M (SD)'], ['Treatment', '61.3']]))
        assert found[('treatment_mean_sd', 2, 2)] == (None, 'review')

    def test_unmapped_header_is_reported_not_guessed(self):
        records = values([['Group', 'Score'], ['Treatment', '61.3']])
        assert by_field(records)[('unmapped', 2, 2)] == (61.3, 'review')

    def test_quote_names_the_row_and_column(self):
        record = values([['Group', 'M'], ['Treatment', '61.3']])[0]
        assert record['quote'] == 'Treatment | M: 61.3'

    def test_pdf_table_keeps_page(self, study_pdf):
        rows = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]
        records = extract_table_values(read_tables(study_pdf(rows)))
        found = {r['field']: r for r in records}
        assert found['control_mean']['value'] == 55.0 and found['control_mean']['pdf_page'] == 1


class TestOutcomeAndArm:
    def test_outcome_column_filled_down_with_arm_column(self):
        records = values([['Outcome', 'Group', 'M'], ['Reading', 'Treatment', '61.3'],
                          ['', 'Control', '55.0'], ['Math', 'Treatment', '48.0'],
                          ['', 'Control', '47.0']])
        found = {(r['outcome'], r['group']): (r['value'], r['field']) for r in records}
        assert found[('Reading', 'control')] == (55.0, 'reading_control_mean')
        assert found[('Math', 'treatment')] == (48.0, 'math_treatment_mean')

    def test_outcome_from_column_label(self):
        records = values([['', 'Reading', '', 'Math', ''], ['Group', 'M', 'SD', 'M', 'SD'],
                          ['Treatment', '61.3', '14.2', '48.0', '9.1']])
        record = next(r for r in records if r['col'] == 4)
        assert identity(record) == ('Math', 'Treatment', 'treatment', None, 'mean')
        assert record['field'] == 'math_treatment_mean'

    def test_time_and_outcome_from_row_labels(self):
        records = values([['Measure', 'Treatment M', 'Control M'], ['Reading pre', '50.1', '50.4'],
                          ['Reading post', '61.3', '55.0'], ['Follow-up', '60.2', '54.8']])
        found = {(r['row'], r['col']): identity(r) for r in records}
        assert found[(2, 2)] == ('Reading', 'Treatment', 'treatment', 'pre', 'mean')
        assert found[(4, 3)] == ('Reading', 'Control', 'control', 'follow_up', 'mean')

    def test_two_treatment_arms_do_not_collide(self):
        records = values([['Group', 'M'], ['Treatment A', '61.3'], ['Treatment B', '58.0'],
                          ['Control', '55.0']])
        found = {r['field']: (r['arm'], r['group'], r['value']) for r in records}
        assert found['treatment_a_mean'] == ('Treatment A', 'treatment', 61.3)
        assert found['treatment_b_mean'] == ('Treatment B', 'treatment', 58.0)
        assert found['control_mean'] == ('Control', 'control', 55.0)
        assert set(to_wide(records)[0]) == {'paper_id', 'treatment_a_mean', 'treatment_b_mean',
                                            'control_mean'}


class TestVetoAndAdjusted:
    def test_differences_and_intervals_are_never_group_stats(self):
        records = values([['Group', 'M', '95% CI', 'Mean difference', 'Mean change'],
                          ['Treatment', '61.3', '58.1', '6.3', '4.0']])
        stats = {r['col']: (r['stat'], r['field'], r['status']) for r in records}
        assert stats[3] == stats[4] == stats[5] == (None, 'unmapped', 'review')

    def test_vetoed_pair_cells_are_not_read_as_mean_and_sd(self):
        records = values([['Group', 'Mean difference (SE)'], ['Treatment', '6.3 (1.1)']])
        assert [(r['field'], r['value']) for r in records] == [('unmapped', 6.3)]

    def test_adjusted_mean_with_se_keeps_both_at_posttest(self):
        found = by_field(values([['Group', 'Adj. M (SE)'], ['Treatment', '60.1 (1.2)']]))
        assert found[('treatment_post_adj_mean', 2, 2)] == (60.1, 'extracted')
        assert found[('treatment_post_se', 2, 2)] == (1.2, 'extracted')


ARM_COLUMNS = [['Outcome', 'Treatment', 'Control'], ['Reading', '61.3 (14.2)', '55.0 (13.1)']]
COUNTS = [['Characteristic', 'Treatment', 'Control'], ['Female', '12 (40%)', '15 (50%)']]


class TestCellShape:
    def test_shape_without_stated_convention_needs_review(self):
        records = values(ARM_COLUMNS)
        found = by_field(records)
        assert found[('treatment_mean', 2, 2)] == (61.3, 'review')
        assert found[('control_sd', 2, 3)] == (13.1, 'review')

    def test_convention_in_header_rows_settles_the_shape(self):
        rows = [['Outcome, M (SD)', 'Treatment', 'Control'], ARM_COLUMNS[1]]
        assert by_field(values(rows))[('treatment_sd', 2, 2)] == (14.2, 'extracted')

    def test_convention_in_read_note_settles_the_shape(self):
        grid = table_grid({'table': 1, 'rows': COUNTS, 'note': 'Note. Values are n (%).'})
        found = by_field(extract_table_values([grid]))
        assert found[('treatment_n', 2, 2)] == (12, 'extracted')
        assert found[('control_percent', 2, 3)] == (50.0, 'extracted')

    def test_convention_in_caption_settles_the_shape(self):
        grid = table_grid({'table': 1, 'rows': ARM_COLUMNS, 'caption': 'Table 1 Means (SD)'})
        assert by_field(extract_table_values([grid]))[('treatment_mean', 2, 2)][1] == 'extracted'

    def test_notes_argument_replaces_the_read_note(self):
        grid = table_grid({'table': 1, 'rows': COUNTS, 'note': 'Note. Values are n (%).'})
        found = by_field(extract_table_values([grid], notes={1: 'Values are M (SD).'}))
        assert found[('treatment_n', 2, 2)] == (12, 'review')

    def test_plus_minus_cells_read_as_mean_and_sd(self):
        rows = [['Outcome', 'Treatment'], ['Reading', '61.3 ± 14.2']]
        found = by_field(values(rows, notes={1: 'Mean ± SD'}))
        assert found[('treatment_mean', 2, 2)] == (61.3, 'extracted')


class TestNothingSilent:
    def test_table_without_numbers_gives_a_flagged_record(self):
        records = values([['Study', 'Design'], ['A', 'RCT']])
        assert [(r['field'], r['status'], r['quote'], r['table']) for r in records] == [
            ('table', 'not_found', 'table has no numeric cells', 1)]

    def test_numbers_that_give_no_values_are_flagged_for_review(self):
        records = values([['Year', 'Design'], ['2019', 'RCT'], ['2021', 'QED']])
        assert [(r['field'], r['status']) for r in records] == [('table', 'review')]

    def test_document_without_tables_gives_a_flagged_record(self):
        records = extract_table_values([], paper_id='S01')
        assert [(r['paper_id'], r['field'], r['status'], r['quote']) for r in records] == [
            ('S01', 'table', 'not_found', 'no tables found')]

    def test_missing_permission_is_flagged_not_raised(self):
        records = values([['Group', 'Score'], ['Treatment', '61.3']],
                         respond=lambda prompt, images=None: '{}')
        record = one(records, 'unmapped')
        assert (record['value'], record['status'], record['layer']) == (61.3, 'refused', 'table')
        assert 'permission is required' in record['quote']

    def test_forbidding_permission_refuses_model_columns_and_sends_nothing(self):
        calls = []
        refused = check_llm_use(access='subscription', ai_terms='forbids')
        records = values([['Group', 'Score', 'M'], ['Treatment', '61.3', '50.0']],
                         respond=lambda prompt, images=None: calls.append(prompt) or '{}',
                         permission=refused, endpoint='http://localhost:11434/v1')
        found = {r['col']: r for r in records}
        assert calls == []
        assert (found[2]['status'], found[2]['layer'], 'forbid' in found[2]['quote']) == (
            'refused', 'table', True)
        assert (found[3]['field'], found[3]['status']) == ('treatment_mean', 'extracted')

    def test_refused_group_labels_flag_their_cells(self):
        records = values([['Group', 'M'], ['Peer tutoring', '61.3']],
                         respond=lambda prompt, images=None: '{}')
        record = one(records, 'peer_tutoring_mean')
        assert record['status'] == 'refused' and 'permission is required' in record['quote']

    def test_no_respond_means_no_refusal(self):
        assert one(values([['Group', 'Score'], ['Treatment', '61.3']]), 'unmapped')[
            'status'] == 'review'


UNMAPPED = [['Outcome', 'Treatment (n = 30)', 'Control (n = 28)', '95% CI'],
            ['Reading', '12.3 (4.5)a', '\u2014', '[0.5, 3.9]'],
            ['Math', 'n.r.', 'abc', 'not reported']]


class TestEveryCellCounted:
    def test_dash_and_text_cells_under_unmapped_headers_give_records(self):
        found = {(r['row'], r['col']): r for r in values(UNMAPPED) if r['row'] != 1}
        assert (found[(2, 3)]['status'], found[(2, 3)]['arm'], found[(2, 3)]['outcome']) == (
            'not_found', 'Control', 'Reading')
        assert (found[(3, 2)]['status'], found[(3, 2)]['arm'], found[(3, 2)]['outcome']) == (
            'not_found', 'Treatment', 'Math')
        assert (found[(3, 3)]['status'], found[(3, 3)]['value'], found[(3, 3)]['quote']) == (
            'review', None, 'abc')

    def test_interval_under_a_vetoed_column_is_unmapped_review(self):
        record = next(r for r in values(UNMAPPED) if (r['row'], r['col']) == (2, 4))
        assert (record['field'], record['status'], record['value']) == ('unmapped', 'review', None)
        assert record['quote'] == '[0.5, 3.9]'

    def test_every_body_cell_has_a_record(self):
        records = values(UNMAPPED)
        cells = {(r['row'], r['col']) for r in records if r['row'] and r['row'] > 1}
        assert cells >= {(r, c) for r in (2, 3) for c in (2, 3, 4)}
        assert len(records) - len([r for r in records if r['row'] == 1]) >= 6


class TestLayers:
    def test_model_mapping_with_exact_quote_is_extracted_as_prompt(self):
        reply = {'labels': [{'number': 1, 'choice': 'mean', 'quote': 'Score'}]}
        records = values([['Group', 'Score'], ['Treatment', '61.3']],
                         respond=lambda prompt, images=None: json.dumps(reply), **LOCAL)
        record = one(records, 'treatment_mean')
        assert (record['value'], record['status'], record['layer']) == (61.3, 'extracted', 'prompt')

    def test_human_override_gives_layer_human(self):
        records = values([['Group', 'Score'], ['Arm A', '61.3']],
                         header_overrides={1: {'Score': 'mean'}},
                         group_overrides={1: {'Arm A': 'treatment'}})
        record = one(records, 'treatment_mean')
        assert (record['status'], record['layer'], record['arm']) == ('extracted', 'human', 'Arm A')


class TestStatsAsRows:
    ROWS = [['', 'Treatment', 'Control'], ['n', '39', '41'], ['M', '59.12', '54.34'],
            ['SD', '12.32', '11.57']]

    def test_row_labels_give_stats_and_headers_give_arms(self):
        records = values(self.ROWS)
        assert {(r['group'], r['stat']): (r['value'], r['status']) for r in records} == {
            (group, stat): (value, 'extracted') for group, stat, value in (
                ('treatment', 'n', 39), ('treatment', 'mean', 59.12), ('treatment', 'sd', 12.32),
                ('control', 'n', 41), ('control', 'mean', 54.34), ('control', 'sd', 11.57))}
        assert [r['row'] for r in records if r['stat'] == 'mean'] == [3, 3]

    def test_effect_matches_groups_as_rows(self):
        from pubextractor.comparisons import build_comparisons
        from pubextractor.effectsize import effect_sizes
        flipped = effect_sizes(build_comparisons(values(self.ROWS),
                                                 studies={'S01': {'design': 'rct'}}))
        upright = effect_sizes(build_comparisons(
            values([['Group', 'n', 'M', 'SD'], ['Treatment', '39', '59.12', '12.32'],
                    ['Control', '41', '54.34', '11.57']]), studies={'S01': {'design': 'rct'}}))
        assert [(e['status'], round(e['g'], 4)) for e in flipped] == [('computed', 0.3964)]
        assert flipped[0]['g'] == upright[0]['g']

    def test_timed_row_labels_keep_their_time(self):
        records = values([['', 'Treatment', 'Control'], ['Pre M', '50', '50'],
                          ['Post M', '58', '54']])
        assert {(r['group'], r['time'], r['stat']) for r in records} == {
            ('treatment', 'pre', 'mean'), ('control', 'pre', 'mean'),
            ('treatment', 'post', 'mean'), ('control', 'post', 'mean')}


SUBHEADED = [['Group', 'n', 'Pretest M (SD)', 'Posttest M (SD)'],
             ['Reading', '', '', ''],
             ['Treatment', '14', '15.4 (5.8)', '21.4 (5.9)'],
             ['Control', '15', '19.5 (8.0)', '21.9 (6.6)'],
             ['Spelling', '—', '', '—'],
             ['Treatment', '14', '1.2 (1.5)', '4.3 (3.2)'],
             ['Control', '15', '1.8 (2.2)', '14.7 (12.8)']]
SUBHEADED_VALUES = {
    (outcome, arm, time, stat, value)
    for outcome, rows in (('Reading', ((14, 15.4, 5.8, 21.4, 5.9), (15, 19.5, 8.0, 21.9, 6.6))),
                          ('Spelling', ((14, 1.2, 1.5, 4.3, 3.2), (15, 1.8, 2.2, 14.7, 12.8))))
    for arm, (n, pre_m, pre_sd, post_m, post_sd) in zip(('Treatment', 'Control'), rows)
    for time, stat, value in ((None, 'n', n), ('pre', 'mean', pre_m), ('pre', 'sd', pre_sd),
                              ('post', 'mean', post_m), ('post', 'sd', post_sd))}


def subheaded(records):
    return {(r['outcome'], r['arm'], r['time'], r['stat'], r['value']) for r in records}


class TestOutcomeSubheaders:
    def test_subheader_rows_set_outcome_and_are_not_arms(self):
        records = values(SUBHEADED)
        assert subheaded(records) == SUBHEADED_VALUES
        assert {r['status'] for r in records} == {'extracted'}
        assert {r['row'] for r in records} == {3, 4, 6, 7}
        assert one(records, 'spelling_control_post_mean')['value'] == 14.7

    def test_subheader_label_away_from_the_stub(self):
        rows = [SUBHEADED[0], ['', '', 'Reading', ''], *SUBHEADED[2:4], ['', '', 'Spelling', ''],
                *SUBHEADED[5:]]
        assert subheaded(values(rows)) == SUBHEADED_VALUES

    def test_subheader_under_two_row_header(self):
        rows = [['', '', 'Pretest', 'Posttest'], ['Group', 'n', 'M (SD)', 'M (SD)']] + SUBHEADED[1:]
        records = values(rows)
        assert subheaded(records) == SUBHEADED_VALUES
        assert {r['row'] for r in records} == {4, 5, 7, 8}

    def test_docx_subheaders(self, tmp_path):
        docx = pytest.importorskip('docx')
        document = docx.Document()
        document.add_paragraph('Table 1')
        table = document.add_table(rows=len(SUBHEADED), cols=4)
        for r, row in enumerate(SUBHEADED):
            for c, text in enumerate(row):
                table.cell(r, c).text = text
        path = tmp_path / 'subheaded.docx'
        document.save(path)
        assert subheaded(extract_table_values(read_tables(path))) == SUBHEADED_VALUES

    def test_pdf_subheaders(self, study_pdf):
        records = extract_table_values(read_tables(study_pdf(SUBHEADED)))
        assert subheaded(records) == SUBHEADED_VALUES
        assert {r['status'] for r in records} == {'extracted'}

    def test_lone_row_without_numbers_below_stays_an_arm(self):
        records = values([['Group', 'M'], ['Treatment', '61.3'], ['Control', '—']])
        assert identity(one(records, 'control_mean')) == (None, 'Control', 'control', None, 'mean')


class TestOutputColumns:
    def test_columns_carry_table_position_and_identity(self):
        assert COLUMNS[COLUMNS.index('table'):][:3] == ('table', 'row', 'col')
        assert {'group', 'arm', 'time', 'stat', 'outcome'} <= set(COLUMNS)
        assert 'table_id' not in COLUMNS

    def test_long_and_wide_take_table_records(self):
        records = values([['Group', 'M'], ['Treatment', '61.3'], ['Control', '55.0']])
        assert to_long(records)[0]['stat'] == 'mean' and to_long(records)[0]['row'] == 2
        assert to_wide(records) == [{'paper_id': 'S01', 'treatment_mean': '61.3',
                                     'control_mean': '55.0'}]

    def test_layer_records_get_empty_table_columns(self):
        record = {'paper_id': 'S01', 'field': 'design', 'value': 'group', 'layer': 'regex'}
        row = to_long([record])[0]
        assert (row['group'], row['arm'], row['stat'], row['outcome'], row['row']) == (
            None, None, None, None, None)


def test_package_exports_plan_3_names():
    import pubextractor

    for name in ('read_tables', 'table_grid', 'table_pages', 'read_header', 'parse_cell',
                 'header_n', 'STAT_ROLES', 'GROUP_ROLES', 'map_headers', 'find_groups',
                 'extract_table_values'):
        assert name in pubextractor.__all__ and hasattr(pubextractor, name)
