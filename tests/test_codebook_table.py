import csv
from pathlib import Path

import pytest

from pubextractor import load_codebook, read_codebook_table, write_codebook_table
from pubextractor.codebook_table import COLUMNS

FIELDS = {
    'design': {'type': 'category', 'values': ['group', 'single_case'],
               'definition': 'Research design of the study', 'color': '#1f77b4',
               'terms': {'group': ['randomly assigned', 'control group'],
                         'single_case': ['multiple baseline']}},
    'grade_band': {'type': 'multi_category', 'values': ['elementary', 'middle', 'high'],
                   'terms': {'middle': ['middle school']}},
    'sample_size': {'type': 'integer', 'stat': 'n', 'range': [1, 100000],
                    'definition': 'Analytic sample'},
    'effect': {'type': 'number', 'range': [-3.5, 3.5]},
    'setting': {'type': 'text', 'query': 'Where did the study take place?', 'color': '#2ca02c'},
}

HEADER = ','.join(COLUMNS)
EXAMPLE = Path(__file__).resolve().parents[1] / 'pubextractor' / 'examples' / 'codebook.xlsx'


def _csv(tmp_path, *lines, header=HEADER):
    path = tmp_path / 'codebook.csv'
    path.write_text('\n'.join([header, *lines]) + '\n', encoding='utf-8')
    return path


class TestRoundTrip:
    @pytest.mark.parametrize('suffix', ['.xlsx', '.csv'])
    def test_write_then_read_gives_the_same_codebook(self, tmp_path, suffix):
        if suffix == '.xlsx':
            pytest.importorskip('openpyxl')
        book = load_codebook(FIELDS)
        path = write_codebook_table(book, tmp_path / f'codebook{suffix}')
        assert read_codebook_table(path) == book

    def test_long_format_has_one_row_per_term(self, tmp_path):
        path = write_codebook_table(load_codebook(FIELDS), tmp_path / 'codebook.csv')
        with open(path, encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0]) == list(COLUMNS)
        design = [(r['value'], r['term']) for r in rows if r['field'] == 'design']
        assert design == [('group', 'randomly assigned'), ('group', 'control group'),
                          ('single_case', 'multiple baseline')]
        grade = [(r['value'], r['term']) for r in rows if r['field'] == 'grade_band']
        assert grade == [('elementary', ''), ('middle', 'middle school'), ('high', '')]
        assert [r['term'] for r in rows if r['field'] == 'sample_size'] == ['']
        assert {r['color'] for r in rows if r['field'] == 'design'} == {'#1f77b4'}


class TestRead:
    def test_selected_false_drops_the_term(self, tmp_path):
        path = _csv(tmp_path,
                    'design,category,group,randomly assigned,,,,,,TRUE,',
                    'design,category,group,cohort,,,,,,FALSE,',
                    'design,category,single_case,,,,,,,,')
        field = read_codebook_table(path)['design']
        assert field['values'] == ['group', 'single_case']
        assert field['terms'] == {'group': ['randomly assigned']}

    def test_optional_columns_may_be_absent(self, tmp_path):
        path = _csv(tmp_path, 'design,category,group,control group',
                    header='field,type,value,term')
        field = read_codebook_table(path)['design']
        assert field['terms'] == {'group': ['control group']} and field['color'] is None

    def test_numeric_cells_become_typed(self, tmp_path):
        path = _csv(tmp_path, 'sample_size,integer,,,,,n,1,500,,')
        assert read_codebook_table(path)['sample_size']['range'] == (1, 500)

    def test_xlsx_booleans_and_numbers(self, tmp_path):
        openpyxl = pytest.importorskip('openpyxl')
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(list(COLUMNS))
        sheet.append(['design', 'category', 'group', 'control group', None, None, None, None,
                      None, True, '#d62728'])
        sheet.append(['design', 'category', 'group', 'cohort', None, None, None, None,
                      None, False, None])
        sheet.append(['design', 'category', 'single_case', None, None, None, None, None,
                      None, None, None])
        sheet.append(['sample_size', 'integer', None, None, None, None, 'n', 1, 100000.0,
                      None, None])
        path = tmp_path / 'codebook.xlsx'
        book.save(path)
        found = read_codebook_table(path)
        assert found['design']['terms'] == {'group': ['control group']}
        assert found['design']['color'] == '#d62728'
        assert found['sample_size']['range'] == (1, 100000)

    def test_example_codebook_loads(self):
        pytest.importorskip('openpyxl')
        found = read_codebook_table(EXAMPLE)
        assert list(found) == ['design', 'sample_size', 'grade_band', 'setting',
                               'outcome_domain']
        assert found['outcome_domain']['values'] == ['achievement', 'behavioural']
        assert found['setting']['query']


class TestProblems:
    @pytest.mark.parametrize('line, message', [
        (',category,group,,,,,,,,', 'row 2: field is empty'),
        ('design,category,,,,,,,,,', 'row 2: design: value is empty'),
        ('design,category,group,,,,,,,maybe,', 'row 2: selected must be TRUE or FALSE'),
        ('n,integer,,,,,n,one,10,,', 'row 2: range_min must be a number'),
        ('n,integer,,,,,n,1,,,', 'row 2: range needs both range_min and range_max'),
        ('n,integer,5,,,,n,,,,', 'row 2: n: value must be empty for integer fields'),
    ])
    def test_row_problems_name_the_sheet_row(self, tmp_path, line, message):
        with pytest.raises(ValueError, match=message):
            read_codebook_table(_csv(tmp_path, line))

    def test_type_conflict_names_the_row(self, tmp_path):
        path = _csv(tmp_path, 'design,category,group,,,,,,,,', 'design,text,,,,,,,,,')
        with pytest.raises(ValueError, match='row 3: design: type text differs from row 2'):
            read_codebook_table(path)

    def test_codebook_problems_name_the_first_row_of_the_field(self, tmp_path):
        path = _csv(tmp_path, 'design,category,group,,,,,,,,', 'size,color,,,,,,,,,')
        with pytest.raises(ValueError, match=r"row 3: size: type must be"):
            read_codebook_table(path)

    def test_bad_color_is_a_problem(self, tmp_path):
        with pytest.raises(ValueError, match='row 2: design: color must be'):
            read_codebook_table(_csv(tmp_path, 'design,category,group,,,,,,,,blue'))

    def test_missing_columns_are_named(self, tmp_path):
        with pytest.raises(ValueError, match="missing columns: type"):
            read_codebook_table(_csv(tmp_path, 'design,group', header='field,value'))

    def test_unknown_suffix(self, tmp_path):
        path = tmp_path / 'codebook.txt'
        path.write_text('field,type\n', encoding='utf-8')
        with pytest.raises(ValueError, match='.xlsx or .csv'):
            read_codebook_table(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_codebook_table(tmp_path / 'absent.xlsx')

    def test_empty_sheet(self, tmp_path):
        with pytest.raises(ValueError, match='codebook has no fields'):
            read_codebook_table(_csv(tmp_path))


class TestColor:
    def test_load_codebook_keeps_a_hex_color(self):
        book = load_codebook({'a': {'type': 'text', 'color': '#AbCdEf'}})
        assert book['a']['color'] == '#AbCdEf'

    def test_color_defaults_to_none(self):
        assert load_codebook({'a': {'type': 'text'}})['a']['color'] is None

    @pytest.mark.parametrize('color', ['red', '#12345', 12])
    def test_load_codebook_rejects_other_colors(self, color):
        with pytest.raises(ValueError, match='color must be'):
            load_codebook({'a': {'type': 'text', 'color': color}})


class TestTableHardening:
    def test_type_is_trimmed_and_lower_cased(self, tmp_path):
        path = _csv(tmp_path, 'design, Category ,group,randomly assigned,,,,,,,',
                    'sample,  INTEGER ,,,,,,,,,')
        assert read_codebook_table(path)['design']['type'] == 'category'
        assert read_codebook_table(path)['sample']['type'] == 'integer'

    def test_blank_type_gives_one_message(self, tmp_path):
        path = _csv(tmp_path, 'design,,group,randomly assigned,,,,,,,')
        with pytest.raises(ValueError) as caught:
            read_codebook_table(path)
        assert str(caught.value).count('row 2:') == 1 and 'row 2: type is empty' in str(caught.value)

    def test_duplicate_row_is_dropped_with_a_warning(self, tmp_path, caplog):
        path = _csv(tmp_path, 'design,category,group,randomly assigned,,,,,,,',
                    'design,category,group,randomly assigned,,,,,,,')
        with caplog.at_level('WARNING', logger='pubextractor.codebook_table'):
            book = read_codebook_table(path)
        assert book['design']['terms'] == {'group': ['randomly assigned']}
        assert 'row 3' in caplog.text and 'duplicate' in caplog.text

    def test_corrupt_workbook_raises_value_error(self, tmp_path):
        pytest.importorskip('openpyxl')
        path = tmp_path / 'broken.xlsx'
        path.write_bytes(b'this is not a zip file')
        with pytest.raises(ValueError, match='could not read broken.xlsx'):
            read_codebook_table(path)


class TestSafeColor:
    @pytest.mark.parametrize('value', ['#1f77b4', '#ABCDEF'])
    def test_valid_hex_is_returned(self, value):
        from pubextractor.codebook import safe_color

        assert safe_color(value, '#000000') == value

    @pytest.mark.parametrize('value', [None, '', 'red', '#12345', '#12345g', 12,
                                       '#1f77b4;x', '#1f77b4\n', 'url(#1f77b4)'])
    def test_anything_else_gives_the_default(self, value):
        from pubextractor.codebook import safe_color

        assert safe_color(value, '#2ca02c') == '#2ca02c'
