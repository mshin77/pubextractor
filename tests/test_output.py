import csv

from pubextractor.output import COLUMNS, to_long, to_wide, write_csv

RECORDS = [
    {'paper_id': 'S01', 'field': 'grade_band', 'value': 'middle', 'layer': 'regex',
     'status': 'extracted', 'quote': 'middle school', 'pdf_page': 1, 'printed_page': 101,
     'section': 'Method', 'candidates': 1},
    {'paper_id': 'S01', 'field': 'data_collection', 'value': 'writing', 'layer': 'regex',
     'status': 'extracted', 'quote': 'journals', 'pdf_page': 2, 'printed_page': 102,
     'section': 'Results', 'candidates': 2},
    {'paper_id': 'S01', 'field': 'data_collection', 'value': 'observation', 'layer': 'regex',
     'status': 'extracted', 'quote': 'observed', 'pdf_page': 2, 'printed_page': 102,
     'section': 'Results', 'candidates': 2},
    {'paper_id': 'S02', 'field': 'grade_band', 'value': None, 'layer': 'none',
     'status': 'not_found', 'quote': '', 'pdf_page': None, 'printed_page': None,
     'section': '', 'candidates': 0},
]


class TestOutput:
    def test_long_keeps_every_record_in_column_order(self):
        rows = to_long(RECORDS)
        assert len(rows) == 4 and list(rows[0]) == list(COLUMNS)

    def test_wide_one_row_per_paper(self):
        rows = to_wide(RECORDS)
        assert rows == [{'paper_id': 'S01', 'grade_band': 'middle',
                         'data_collection': 'writing; observation'},
                        {'paper_id': 'S02', 'grade_band': '', 'grade_band_status': 'not_found'}]

    def test_wide_keeps_review_and_refused_visible(self):
        extra = [
            {'paper_id': 'S03', 'field': 'sample_size', 'value': 48, 'status': 'review'},
            {'paper_id': 'S03', 'field': 'sample_size', 'value': 50, 'status': 'review'},
            {'paper_id': 'S03', 'field': 'setting', 'value': None, 'status': 'refused'},
            {'paper_id': 'S03', 'field': 'design', 'value': 'group', 'status': 'extracted'},
        ]
        assert to_wide(extra) == [{'paper_id': 'S03', 'sample_size': '48; 50',
                                   'sample_size_status': 'review', 'setting': '',
                                   'setting_status': 'refused', 'design': 'group'}]

    def test_csv_round_trip(self, tmp_path):
        path = tmp_path / 'long.csv'
        write_csv(to_long(RECORDS), path)
        with open(path, encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]['field'] == 'grade_band' and rows[0]['printed_page'] == '101'
        assert list(rows[0]) == list(COLUMNS)

    def test_csv_quotes_commas_newlines_and_non_ascii(self, tmp_path):
        row = {'paper_id': 'S1', 'quote': 'a, "b"\nc \u00e9'}
        path = tmp_path / 'q.csv'
        write_csv([row, {'paper_id': 'S2', 'extra': 'x'}], path)
        with open(path, encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]['quote'] == row['quote'] and rows[1]['extra'] == 'x'


def test_package_exports_plan_2_names():
    import pubextractor

    for name in ('load_codebook', 'import_extraction_schema', 'find_statistics',
                 'extract_values', 'to_long', 'to_wide', 'write_csv'):
        assert name in pubextractor.__all__ and hasattr(pubextractor, name)
