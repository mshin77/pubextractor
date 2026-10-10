import pytest

from pubextractor.codebook import load_codebook
from pubextractor.layers import extract_values
from pubextractor.tables import (
    extract_table_values,
    read_header,
    read_tables,
    table_grid,
    table_pages,
)

GROUP_ROWS = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]


class TestTableGrid:
    def test_docx_table_keeps_location_and_merged_cells_once(self, study_docx):
        grid = read_tables(study_docx)[0]
        assert (grid['table'], grid['section'], grid['pdf_page']) == (1, 'Results', None)
        assert grid['rows'][0] == ['', 'Treatment (n = 24)', '', 'Control (n = 22)', '']
        assert {'row': 3, 'col': 2, 'text': '61.3'} in grid['cells']

    def test_docx_caption_and_note(self, study_docx):
        grid = read_tables(study_docx)[0]
        assert grid['caption'] == 'Table 1 Reading scores by group'
        assert grid['note'] == 'Note. Higher scores mean better reading.'

    def test_pdf_table_keeps_page(self, study_pdf):
        grid = read_tables(study_pdf(GROUP_ROWS))[0]
        assert (grid['table'], grid['pdf_page']) == (1, 1)
        assert grid['rows'] == GROUP_ROWS
        assert (grid['caption'], grid['note']) == ('', '')

    def test_pdf_caption_above_and_note_below(self, study_pdf):
        path = study_pdf(GROUP_ROWS, caption=['Table 2', 'Posttest scores'],
                         note=['Note. Values are M (SD).'])
        grid = read_tables(path)[0]
        assert grid['caption'] == 'Table 2 Posttest scores'
        assert grid['note'] == 'Note. Values are M (SD).'

    def test_ragged_rows_are_padded_and_whitespace_collapsed(self):
        grid = table_grid({'table': 2, 'rows': [['a', ' b\n c ', None], ['1']]})
        assert grid['rows'] == [['a', 'b c', ''], ['1', '', '']]
        assert len(grid['cells']) == 6 and grid['table'] == 2
        assert (grid['caption'], grid['note']) == ('', '')

    def test_other_files_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match='.pdf, .docx, or image'):
            read_tables(tmp_path / 'study.txt')


def test_docx_tables_become_searchable(study_docx):
    book = load_codebook({'treatment_n': {'type': 'integer', 'stat': 'n', 'range': [1, 1000]}})
    pages = table_pages(read_tables(study_docx))
    record = extract_values(pages, book)[0]
    assert (record['value'], record['table'], record['pdf_page']) == (24, 1, None)
    assert record['section'] == 'Results' and record['status'] == 'review'
    assert pages[0]['text'].startswith('Table 1 Reading scores by group')


TWO_ROW = [['', 'Treatment (n = 24)', '', 'Control (n = 22)', ''],
           ['Measure', 'M', 'SD', 'M', 'SD'],
           ['Reading', '61.3', '14.2', '55.0', '13.1']]


def grid(rows):
    return table_grid({'table': 1, 'rows': rows})


def labels(header):
    return [column['label'] for column in header['columns']]


class TestReadHeader:
    def test_two_header_rows_are_joined_with_group_filled_right(self):
        header = read_header(grid(TWO_ROW))
        assert header['header_rows'] == 2
        assert labels(header) == ['Measure', 'Treatment M', 'Treatment SD',
                                  'Control M', 'Control SD']
        assert header['columns'][2]['parts'] == ['Treatment', 'SD']

    def test_n_in_header_is_read_with_its_row(self):
        columns = read_header(grid(TWO_ROW))['columns']
        assert [c['n'] for c in columns] == [None, 24, 24, 22, 22]
        assert columns[3]['n_row'] == 1 and columns[0]['n_row'] is None

    def test_single_header_row(self):
        header = read_header(grid([['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2']]))
        assert header['header_rows'] == 1 and labels(header) == ['Group', 'M', 'SD']

    def test_blank_cells_in_the_last_header_row_stay_blank(self):
        rows = [['', 'Pre', '', 'Post', ''], ['', 'M', '', 'M', 'SD'],
                ['Reading', '1', '2', '3', '4']]
        assert labels(read_header(grid(rows))) == ['', 'Pre M', 'Pre', 'Post M', 'Post SD']

    def test_years_in_header_cells_do_not_end_the_header(self):
        rows = [['', '2019', '', '2020', ''], ['Group', 'M', 'SD', 'M', 'SD'],
                ['Treatment', '61.3', '14.2', '63.0', '13.9']]
        header = read_header(grid(rows))
        assert header['header_rows'] == 2 and labels(header)[1] == '2019 M'

    def test_year_only_header_row(self):
        header = read_header(grid([['Group', '2019', '2020'], ['Treatment', '61.3', '63.0']]))
        assert header['header_rows'] == 1 and labels(header) == ['Group', '2019', '2020']

    def test_numeric_first_row_has_no_header(self):
        header = read_header(grid([['1', '2'], ['3', '4']]))
        assert header['header_rows'] == 0 and labels(header) == ['', '']

    def test_text_only_table_has_one_header_row(self):
        assert read_header(grid([['a', 'b'], ['c', 'd']]))['header_rows'] == 1

    def test_combined_cells_count_as_numeric(self):
        header = read_header(grid([['Group', 'M (SD)'], ['Treatment', '12.3 (4.5)']]))
        assert header['header_rows'] == 1

    def test_header_rows_can_be_set(self):
        header = read_header(grid(TWO_ROW), header_rows=1)
        assert header['header_rows'] == 1 and labels(header)[1] == 'Treatment'
        assert header['columns'][1]['n'] == 24



PROSE = ('Scores in the tutored group rose by about six points over the term, '
         'and the gain held at the follow-up visit eight weeks later.')
SPANNED = [['', 'Pretest', '', '', 'Posttest', '', ''],
           ['Group', 'n', 'M', 'SD', 'n', 'M', 'SD'],
           ['Tutored', '20', '41.2', '6.1', '19', '48.5', '6.6'],
           ['Control', '21', '40.8', '5.9', '21', '43.0', '6.2']]
CENTERED = [('SPAN', (1, 0), (3, 0)), ('SPAN', (4, 0), (6, 0)), ('ALIGN', (1, 0), (-1, 0), 'CENTER')]


class TestSpanningHeaders:
    def test_centered_label_covers_its_columns(self, borderless_pdf):
        found = read_tables(borderless_pdf(['Table 1', SPANNED, PROSE], table_style=CENTERED))
        labels = [c['label'] for c in read_header(found[0])['columns']]
        assert labels == ['Group', 'Pretest n', 'Pretest M', 'Pretest SD',
                          'Posttest n', 'Posttest M', 'Posttest SD']

    def test_centered_label_sets_time_of_every_value(self, borderless_pdf):
        found = read_tables(borderless_pdf(['Table 1', SPANNED, PROSE], table_style=CENTERED))
        fields = {r['field'] for r in extract_table_values(found, paper_id='S01')
                  if r['field'].startswith('tutored')}
        assert fields == {'tutored_pre_n', 'tutored_pre_mean', 'tutored_pre_sd',
                          'tutored_post_n', 'tutored_post_mean', 'tutored_post_sd'}


ROWS = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]
WIDE = [['', 'Treatment', '', 'Control', ''], ['Measure', 'M', 'SD', 'M', 'SD'],
        ['Reading', '61.3', '14.2', '55.0', '13.1']]
LINES = ('Scores in the tutored group rose by about six points over the term, '
         'and the gain held at the follow-up visit eight weeks later.')


class TestReadDocument:
    def test_pages_and_grids_match_the_separate_readers(self, borderless_pdf):
        from pubextractor.fulltext import read_pdf
        from pubextractor.tables import read_document, read_tables
        path = borderless_pdf(['Table 2', 'Scores by group', ROWS, LINES, None, LINES, LINES],
                              columns=2)
        assert read_document(path) == (read_pdf(path), read_tables(path))

    def test_words_are_read_once_per_page(self, borderless_pdf, monkeypatch):
        from pdfplumber.page import Page

        from pubextractor.fulltext import X_TOLERANCE
        from pubextractor.tables import read_document
        path = borderless_pdf(['Table 1', ROWS, LINES, 'Table 2', WIDE, LINES])
        calls = []
        original = Page.extract_words
        monkeypatch.setattr(Page, 'extract_words',
                            lambda page, **kw: calls.append((page.page_number, kw))
                            or original(page, **kw))
        pages, grids = read_document(path)
        ours = sorted(number for number, kw in calls if kw == {'x_tolerance': X_TOLERANCE})
        assert len(grids) == 2 and ours == [p['page'] for p in pages]

    def test_docx_gives_blocks_and_grids(self, study_docx):
        from pubextractor.docx_reader import read_docx
        from pubextractor.tables import read_document, read_tables
        assert read_document(study_docx) == (read_docx(study_docx), read_tables(study_docx))
