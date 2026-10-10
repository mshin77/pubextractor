import json

import numpy as np
import pytest

from pubextractor.codebook import load_codebook
from pubextractor.layers import extract_values
from pubextractor.licensing import check_llm_use
from pubextractor.output import COLUMNS, to_wide, write_csv
from pubextractor.rag import extract_with_rag
from pubextractor.tables import extract_table_values

LOCAL = dict(permission=check_llm_use(access='subscription', ai_terms='silent'),
             endpoint='http://localhost:11434/v1')
REASON = 'no extractable text (scanned or empty file)'
BOOK = load_codebook({'region': {'type': 'category', 'values': ['urban', 'rural'],
                                 'terms': {'urban': ['urban']}},
                      'setting': {'type': 'text', 'query': 'Where?'}})
TEXT_PAGES = [{'page': 1, 'text': 'The study took place in two urban school settings.'},
              {'page': 2, 'text': 'Pupils wrote journals.'}]


class CountingEncoder:
    def __init__(self):
        self.document_calls = 0
        self.calls = 0

    def encode(self, texts, batch_size=None, show_progress_bar=None):
        self.calls += 1
        self.document_calls += batch_size is not None
        return np.array([[float('urban' in t or 'Where' in t), 1.0] for t in texts])


def respond(prompt, images=None):
    return json.dumps({'value': 'urban', 'quote': 'two urban school settings', 'passage': 1})


class TestScannedAndEmpty:
    @pytest.mark.parametrize('model', [{}, {'respond': respond, 'encoder': CountingEncoder(), **LOCAL}])
    def test_blank_pages_give_not_found_with_reason(self, model):
        records = extract_values([{'page': 1, 'text': ''}, {'page': 2, 'text': ' \n'}], BOOK, **model)
        assert [(r['field'], r['status'], r['quote']) for r in records] == [
            ('region', 'not_found', REASON), ('setting', 'not_found', REASON)]

    def test_empty_codebook_raises(self):
        with pytest.raises(ValueError, match='codebook has no fields'):
            load_codebook({})
        with pytest.raises(ValueError, match='codebook has no fields'):
            extract_values(TEXT_PAGES, {})

    def test_write_csv_refuses_no_rows(self, tmp_path):
        with pytest.raises(ValueError):
            write_csv([], tmp_path / 'out.csv')
        assert not (tmp_path / 'out.csv').exists()


def rec(paper, field, value, status='extracted', table=None):
    return {'paper_id': paper, 'field': field, 'value': value, 'status': status, 'table': table}


class TestWide:
    def test_not_found_keeps_column_and_status(self):
        rows = to_wide([rec('S1', 'design', 'group'), rec('S1', 'size', None, 'not_found'),
                        rec('S2', 'design', None, 'not_found'), rec('S2', 'size', None, 'not_found')])
        assert rows == [{'paper_id': 'S1', 'design': 'group', 'size': '', 'size_status': 'not_found'},
                        {'paper_id': 'S2', 'design': '', 'design_status': 'not_found',
                         'size': '', 'size_status': 'not_found'}]

    def test_same_field_in_two_tables_stays_apart(self):
        row = to_wide([rec('S1', 'mean', 1.5, table=1), rec('S1', 'mean', 2.5, table=2)])[0]
        assert row == {'paper_id': 'S1', 'mean_t1': '1.5', 'mean_t2': '2.5'}

    def test_table_field_colliding_with_text_field_is_suffixed(self):
        row = to_wide([rec('S1', 'n', 40), rec('S1', 'n', 38, table=3)])[0]
        assert row == {'paper_id': 'S1', 'n': '40', 'n_t3': '38'}

    def test_single_table_field_keeps_plain_name(self):
        assert to_wide([rec('S1', 'mean', 1.5, table=1)]) == [{'paper_id': 'S1', 'mean': '1.5'}]


class TestPermission:
    def test_missing_permission_gives_refused_records(self):
        sent = []
        record = extract_values(TEXT_PAGES, {'setting': BOOK['setting']},
                                respond=lambda p: sent.append(p) or '', encoder=CountingEncoder())[0]
        assert record['status'] == 'refused' and 'permission' in record['quote'] and sent == []

    def test_rule_fields_keep_status_without_permission(self):
        by = {r['field']: r for r in extract_values(TEXT_PAGES, BOOK, respond=respond,
                                                    encoder=CountingEncoder())}
        assert by['region']['status'] == 'extracted' and by['setting']['status'] == 'refused'


class TestCost:
    def test_document_embedded_once_whatever_the_field_count(self):
        one, three = CountingEncoder(), CountingEncoder()
        book = {name: {**BOOK['setting'], 'query': f'Where {name}?'} for name in 'abc'}
        extract_values(TEXT_PAGES, {'a': book['a']}, respond=respond, encoder=one, **LOCAL)
        extract_values(TEXT_PAGES, book, respond=respond, encoder=three, **LOCAL)
        assert one.document_calls == three.document_calls == 1

    def test_license_gate_runs_per_field(self):
        refused = check_llm_use(access='subscription', ai_terms='forbids')
        records = extract_values(TEXT_PAGES, {'a': BOOK['setting'], 'b': BOOK['setting']},
                                 respond=respond, encoder=CountingEncoder(), permission=refused,
                                 endpoint='http://localhost:11434/v1')
        assert [r['status'] for r in records] == ['refused', 'refused']


class TestFailedCall:
    def test_rag_reports_the_exception_type(self):
        def broken(prompt):
            raise RuntimeError('down')
        result = extract_with_rag(TEXT_PAGES, {'q': 'Where?'}, broken, CountingEncoder(),
                                  permission='unchecked')['q']
        assert result['error'] == 'RuntimeError'

    def test_field_record_is_review_with_reason(self):
        def broken(prompt, images=None):
            raise TimeoutError('slow')
        record = extract_values(TEXT_PAGES, {'setting': BOOK['setting']}, respond=broken,
                                encoder=CountingEncoder(), **LOCAL)[0]
        assert (record['status'], record['quote']) == ('review', 'model call failed: TimeoutError')


class TestDocxTables:
    def test_table_block_rows_are_searched(self):
        pages = [{'kind': 'table', 'table': 2, 'section': 'Method',
                  'rows': [['Setting', 'Site'], ['Urban', 'Two schools']]}]
        record = extract_values(pages, {'region': BOOK['region']})[0]
        assert (record['value'], record['table'], record['status']) == ('urban', 2, 'extracted')

    def test_table_block_reaches_the_model(self):
        pages = [{'kind': 'table', 'table': 2, 'rows': [['Where', 'two urban schools']]}]
        record = extract_values(pages, {'setting': BOOK['setting']}, respond=respond,
                                encoder=CountingEncoder(), **LOCAL)[0]
        assert record['table'] == 2


class TestRecordKeys:
    def test_text_and_table_records_share_keys(self):
        grid = {'table': 1, 'pdf_page': 1, 'printed_page': None, 'section': '', 'caption': '',
                'note': '', 'rows': [['', 'M', 'SD'], ['Treatment', '1.0', '0.5']]}
        text_keys = {frozenset(r) for r in extract_values(TEXT_PAGES, BOOK)}
        table_keys = {frozenset(r) for r in extract_table_values([grid])}
        assert text_keys == table_keys == {frozenset(COLUMNS)}


def test_unread_role_is_not_stored():
    book = load_codebook({'a': {'type': 'text', 'role': 'effect_size', 'note': 'n'}})
    assert 'role' not in book['a'] and book['a']['note'] == 'n'
