import pytest

fitz = pytest.importorskip('fitz')
pytest.importorskip('pdfplumber')

from pubextractor.tables import read_pdf_tables  # noqa: E402


def grid_pdf(tmp_path, rows):
    document = fitz.open()
    page = document.new_page()
    x0, y0, width, height = 72, 72, 100, 24
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            rect = fitz.Rect(x0 + c * width, y0 + r * height,
                             x0 + (c + 1) * width, y0 + (r + 1) * height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 7), text, fontsize=10)
    document.new_page().insert_text((72, 72), 'No table here.')
    path = tmp_path / 'table.pdf'
    document.save(path)
    return path


class TestReadPdfTables:
    def test_grid_table_is_read_with_page(self, tmp_path):
        rows = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]
        tables = read_pdf_tables(grid_pdf(tmp_path, rows))
        assert len(tables) == 1
        assert tables[0]['pdf_page'] == 1 and tables[0]['table'] == 1
        assert tables[0]['rows'] == rows

    def test_page_without_tables_adds_nothing(self, tmp_path):
        rows = [['a', 'b'], ['1', '2']]
        assert {t['pdf_page'] for t in read_pdf_tables(grid_pdf(tmp_path, rows))} == {1}


AGREE = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]
PROSE = ('Scores in the tutored group rose by about six points over the term, '
         'and the gain held at the follow-up visit eight weeks later.')
NESTED = [['', 'Treatment', '', 'Control', ''], ['Measure', 'M', 'SD', 'M', 'SD'],
          ['Reading', '61.3', '14.2', '55.0', '13.1'], ['Spelling', '40.2', '9.1', '38.7', '8.8']]


class TestBorderlessTables:
    def test_table_under_caption_is_read_without_rules(self, borderless_pdf):
        from pubextractor.tables import read_tables
        found = read_tables(borderless_pdf(['Table 1', 'Posttest scores by group', AGREE, PROSE]))
        assert len(found) == 1
        assert found[0]['rows'] == AGREE
        assert (found[0]['strategy'], found[0]['columns_agree']) == ('caption', True)
        assert found[0]['caption'] == 'Table 1 Posttest scores by group'

    def test_heading_and_prose_below_end_the_table(self, borderless_pdf):
        from pubextractor.tables import read_tables
        found = read_tables(borderless_pdf(['Table 1', AGREE, 'Discussion', PROSE]))
        assert found[0]['rows'] == AGREE

    def test_table_in_left_column_ignores_right_column_prose(self, borderless_pdf):
        from pubextractor.tables import read_tables
        items = ['Table 2', 'Scores by group', AGREE, PROSE, None, PROSE, PROSE, PROSE]
        found = read_tables(borderless_pdf(items, columns=2))
        assert [table['rows'] for table in found] == [AGREE]
        assert found[0]['caption'] == 'Table 2 Scores by group'

    def test_grouped_header_stays_review(self, borderless_pdf):
        from pubextractor.tables import extract_table_values, read_tables
        found = read_tables(borderless_pdf(['Table 1', NESTED, PROSE]))
        records = extract_table_values(found, paper_id='S01')
        assert found[0]['columns_agree'] is False
        assert {r['status'] for r in records if r['value'] is not None} == {'review'}

    def test_agreeing_columns_extract(self, borderless_pdf):
        from pubextractor.tables import extract_table_values, read_tables
        records = extract_table_values(read_tables(borderless_pdf(['Table 1', AGREE, PROSE])),
                                       paper_id='S01')
        assert {r['status'] for r in records if r['value'] is not None} == {'extracted'}

    def test_caption_without_numeric_rows_adds_no_table(self, borderless_pdf):
        from pubextractor.tables import read_tables
        assert read_tables(borderless_pdf(['Table 1', 'Timeline of the study', PROSE])) == []
