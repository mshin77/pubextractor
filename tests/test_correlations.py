import csv

import numpy as np
import pytest

from pubextractor.correlations import (
    CORRELATION_COLUMNS,
    correlation_matrix,
    correlation_records,
    map_constructs,
    suggest_constructs,
    suggest_mapping,
    to_matrices,
)
from pubextractor.output import write_csv
from pubextractor.tables import read_tables, table_grid

NUMBERED = [['Variable', '1', '2', '3'],
            ['1. Reading', '—', '', ''],
            ['2. Math', '.45**', '—', ''],
            ['3. Science', '.38', '.52*', '—']]

FULL = [['', 'Reading', 'Math', 'Science'],
        ['Reading', '1', '.45', '.38'],
        ['Math', '.45', '1', '.52'],
        ['Science', '.38', '.52', '1']]


def grid(rows, caption='', note=''):
    return table_grid({'table': 1, 'rows': rows, 'caption': caption, 'note': note})


@pytest.fixture
def lower_docx(tmp_path):
    docx = pytest.importorskip('docx')
    document = docx.Document()
    document.add_paragraph('Table 2')
    document.add_paragraph('Correlations among measures')
    table = document.add_table(rows=4, cols=4)
    for r, texts in enumerate(NUMBERED):
        for c, text in enumerate(texts):
            table.cell(r, c).text = text
    document.add_paragraph('Note. N = 120. *p < .05. **p < .01.')
    path = tmp_path / 'corr.docx'
    document.save(path)
    return path


class TestCorrelationMatrix:
    def test_numbered_lower_triangle_from_docx(self, lower_docx):
        result = correlation_matrix(read_tables(lower_docx)[0])
        assert result['labels'] == ['Reading', 'Math', 'Science']
        assert result['layout'] == 'lower' and (result['n'], result['n_source']) == (120, 'table')
        assert result['matrix'][0, 1] == result['matrix'][1, 0] == 0.45
        assert np.allclose(np.diag(result['matrix']), 1)

    def test_full_matrix_from_pdf(self, study_pdf):
        path = study_pdf(FULL, caption=['Table 3', 'Correlations (N = 85)'])
        result = correlation_matrix(read_tables(path)[0])
        assert result['layout'] == 'full' and result['n'] == 85 and result['pdf_page'] == 1
        assert result['matrix'][2, 1] == 0.52

    def test_upper_triangle_is_mirrored(self):
        rows = [['', 'A', 'B'], ['A', '—', '−.30'], ['B', '', '—']]
        result = correlation_matrix(grid(rows), n=40)
        assert result['layout'] == 'upper' and result['matrix'][1, 0] == -0.30
        assert (result['n'], result['n_source']) == (40, 'study')

    def test_caption_n_wins_over_study_n(self):
        result = correlation_matrix(grid(FULL, caption='Table 1 Correlations (N = 1,024)'), n=50)
        assert (result['n'], result['n_source']) == (1024, 'table')

    def test_extra_descriptive_columns_are_ignored(self):
        rows = [row + extra for row, extra in zip(FULL, [['M', 'SD'], ['5', '1'], ['6', '2'],
                                                         ['7', '3']])]
        assert correlation_matrix(grid(rows), n=60)['matrix'].shape == (3, 3)


class TestChecks:
    @pytest.mark.parametrize('rows', [NUMBERED, FULL])
    def test_clean_tables_have_no_checks(self, rows):
        assert correlation_matrix(grid(rows), n=120)['checks'] == []

    def test_non_positive_definite_is_flagged_not_repaired(self):
        rows = [['', 'A', 'B', 'C'], ['A', '1', '.9', '.9'], ['B', '.9', '1', '-.9'],
                ['C', '.9', '-.9', '1']]
        result = correlation_matrix(grid(rows), n=30)
        assert result['checks'] == ['not positive definite'] and result['matrix'][1, 2] == -0.9

    @pytest.mark.parametrize('rows, n, flag', [
        ([['', 'A', 'B'], ['A', '.85', ''], ['B', '.40', '.90']], 30, 'diagonal not 1'),
        ([['', 'A', 'B'], ['A', '1', '.40'], ['B', '.30', '1']], 30, 'not symmetric'),
        ([['', 'A', 'B'], ['A', '—', ''], ['B', '1.4', '—']], 30, '|r| > 1'),
        ([['', 'A', 'B', 'C'], ['A', '—', '', ''], ['B', '.4', '—', ''], ['C', '', '.3', '—']],
         30, 'blank cells'),
        ([['', 'A', 'B'], ['A', '—', ''], ['B', '.40', '—']], None, 'n missing'),
        ([['', 'A'], ['A', '—']], 30, 'fewer than 2 variables'),
    ])
    def test_flags(self, rows, n, flag):
        assert flag in correlation_matrix(grid(rows), n=n)['checks']

    def test_fewer_columns_than_labels_is_not_square_and_not_filled(self):
        rows = [['', 'A', 'B'], ['A', '—', ''], ['B', '.40', '—'], ['C', '.30', '.20']]
        result = correlation_matrix(grid(rows), n=30)
        assert result['checks'][0] == 'not_square' and np.isnan(result['matrix'][2, 2])
        assert np.isnan(result['matrix'][0, 2])
        records = correlation_records([result])
        assert {r['status'] for r in records} == {'review'}

    def test_footnote_markers_leave_labels(self):
        rows = [['', 'A', 'B'], ['A*', '—', ''], ['B†', '.40', '—']]
        assert correlation_matrix(grid(rows), n=30)['labels'] == ['A', 'B']


class TestRecords:
    def test_one_record_per_pair_in_column_order(self, tmp_path):
        records = correlation_records([correlation_matrix(grid(NUMBERED), n=120)], paper_id='S01')
        assert [(r['var_i'], r['var_j'], r['r']) for r in records] == [
            ('Reading', 'Math', 0.45), ('Reading', 'Science', 0.38), ('Math', 'Science', 0.52)]
        assert tuple(records[0]) == CORRELATION_COLUMNS and records[0]['status'] == 'extracted'
        with open(write_csv(records, tmp_path / 'r.csv'), encoding='utf-8') as handle:
            assert next(csv.reader(handle)) == list(CORRELATION_COLUMNS)

    def test_flagged_matrix_gives_review_with_checks(self):
        record = correlation_records([correlation_matrix(grid(NUMBERED))])[0]
        assert (record['status'], record['checks']) == ('review', 'n missing')

    def test_blank_pair_is_not_found_and_empty_table_is_reported(self):
        rows = [['', 'A', 'B', 'C'], ['A', '—', '', ''], ['B', '.4', '—', ''], ['C', '', '.3', '—']]
        records = correlation_records([correlation_matrix(grid(rows), n=30),
                                       correlation_matrix(grid([['', 'A']]), n=30)])
        assert [r['status'] for r in records] == ['review', 'not_found', 'review', 'not_found']
        assert records[-1]['var_i'] is None


class TestMapConstructs:
    def test_exact_then_normalized(self):
        found = map_constructs(['reading', 'Math ', 'Self-efficacy', 'Anxiety'],
                               ['reading', 'math', 'self_efficacy'])
        assert found['mapping'] == {'reading': 'reading', 'Math ': 'math',
                                    'Self-efficacy': 'self_efficacy'}
        assert found['unmapped'] == ['Anxiety']
        assert found['sources'] == {'reading': 'exact', 'Math ': 'normalized',
                                    'Self-efficacy': 'normalized'}

    def test_two_labels_on_one_construct_stay_unmapped(self):
        found = map_constructs(['Math', 'math'], ['math'])
        assert found['mapping'] == {'math': 'math'} and found['unmapped'] == ['Math']
        assert map_constructs(['Math ', 'MATH'], ['math'])['unmapped'] == ['Math ', 'MATH']

    def test_human_override_wins(self):
        found = map_constructs(['Reading', 'Anxiety'], ['reading', 'math_anxiety'],
                               overrides={'Anxiety': 'math_anxiety', 'Reading': None})
        assert found['mapping'] == {'Anxiety': 'math_anxiety'}
        assert found['unmapped'] == ['Reading'] and found['sources'] == {'Anxiety': 'human'}

    def test_bad_overrides_are_reported(self):
        with pytest.raises(ValueError, match="'Mood' is not a label.*'mood' is not a construct"):
            map_constructs(['Reading'], ['reading'], overrides={'Mood': 'mood'})


POPULATION = np.array([[1, .45, .38], [.45, 1, .52], [.38, .52, 1]])


def study(name, labels, shift, n):
    k = len(labels)
    matrix = POPULATION[:k, :k] + shift * (1 - np.eye(k))
    return {'study': name, 'labels': labels, 'matrix': matrix, 'n': n,
            'mapping': map_constructs(labels, ['reading', 'math', 'science'])['mapping']}


@pytest.fixture
def studies():
    return [study('S01', ['Reading', 'Math', 'Science'], 0.00, 120),
            study('S02', ['reading', 'math', 'science'], 0.04, 85),
            study('S03', ['Reading', 'Math'], -0.03, 200),
            study('S04', ['Reading', 'Math', 'Science'], -0.05, 150),
            study('S 05', ['Reading', 'Math', 'Science'], 0.02, 60)]


class TestToMatrices:
    def test_files_matrix_with_na_and_n(self, tmp_path, studies):
        paths = to_matrices(studies, ['reading', 'math', 'science'], tmp_path / 'out')
        assert [p.name for p in paths] == ['S01.csv', 'S02.csv', 'S03.csv', 'S04.csv',
                                           'S_05.csv', 'n.csv']
        rows = list(csv.reader(open(paths[2], encoding='utf-8')))
        assert rows[0] == ['construct', 'reading', 'math', 'science']
        assert rows[1] == ['reading', '1', '0.42', 'NA'] and rows[3] == ['science', 'NA', 'NA', 'NA']
        assert list(csv.reader(open(paths[5], encoding='utf-8')))[3] == ['S03', 'S03', '200']

    @pytest.mark.parametrize('change, message', [
        ({'n': None}, 'S01: n missing'),
        ({'mapping': {'Reading': 'math', 'Math': 'math'}}, 'two labels map to one construct'),
    ])
    def test_problems_are_reported(self, tmp_path, studies, change, message):
        with pytest.raises(ValueError, match=message):
            to_matrices([{**studies[0], **change}], ['reading', 'math', 'science'], tmp_path)


def test_package_exports_plan_5_names():
    import pubextractor

    for name in ('correlation_matrix', 'correlation_checks', 'correlation_records',
                 'CORRELATION_COLUMNS', 'map_constructs', 'to_matrices'):
        assert name in pubextractor.__all__ and hasattr(pubextractor, name)


def test_mean_and_sd_columns_stay_out_of_the_matrix():
    grid = table_grid({'table': 1, 'caption': 'Table 1. Correlations (N = 90)', 'rows': [
        ['', '1', '2', 'M', 'SD'], ['1. Reading', '—', '', '50.2', '9.1'],
        ['2. Math', '.42', '—', '40.3', '8.0'], ['3. Science', '.31', '.28', '30.1', '7.2']]})
    found = correlation_matrix(grid)
    assert found['checks'] == []
    assert found['matrix'][2, 2] == 1 and found['matrix'][0, 2] == pytest.approx(0.31)


def test_matrix_that_is_not_a_correlation_is_refused(tmp_path):
    study = {'study': 'S1', 'labels': ['a', 'b'], 'matrix': [[1, 50.2], [50.2, 1]], 'n': 90,
             'mapping': {'a': 'a', 'b': 'b'}}
    with pytest.raises(ValueError, match='S1: not a correlation matrix'):
        to_matrices([study], ['a', 'b'], tmp_path)


class TestSuggestConstructs:
    def test_shared_labels_come_first_and_near_spellings_fold(self):
        found = suggest_constructs([['Reading', 'Math achievement', 'Motivation'],
                                    ['reading', 'Mathematics achievement'], ['Reading']])
        assert found == ['Reading', 'Math achievement', 'Motivation']

    def test_mapping_gives_each_label_a_reason(self):
        found = suggest_mapping(['reading', 'Mathematics achievement', 'Anxiety'],
                                ['Reading', 'Math achievement', 'Motivation'])
        assert found == {'reading': ('Reading', 'normalized'),
                         'Mathematics achievement': ('Math achievement', 'similar 0.82'),
                         'Anxiety': (None, 'no match')}

    def test_two_close_constructs_leave_the_label_open(self):
        found = suggest_mapping(['Math score'], ['Math scores', 'Math scored'])
        assert found == {'Math score': (None, 'no match')}
