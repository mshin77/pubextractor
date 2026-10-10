import numpy as np
import pytest

from pubextractor.codebook import load_codebook
from pubextractor.codebook_table import read_codebook_table, write_codebook_table
from pubextractor.comparisons import build_comparisons
from pubextractor.correlations import map_constructs, to_matrices
from pubextractor.effectsize import compute_effect
from pubextractor.evaluate import evaluate, icc, kappa, read_coding_sheet, write_coding_sheet
from pubextractor.layers import extract_values
from pubextractor.licensing import check_llm_use
from pubextractor.output import to_wide
from pubextractor.providers import make_responder
from pubextractor.rag import extract_with_rag, retrieve


class Encoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 2))


@pytest.mark.parametrize('names', [('A B', 'A_B'), ('n',), ('S1', 's1')])
def test_matrix_export_rejects_filename_collisions(tmp_path, names):
    studies = [{'study': name, 'labels': ['a', 'b'], 'matrix': np.eye(2), 'n': 20,
                'mapping': {'a': 'a', 'b': 'b'}} for name in names]
    with pytest.raises(ValueError, match='file'):
        to_matrices(studies, ['a', 'b'], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('fields', [
    [('paper_id', None, 'extracted')],
    [('a', None, 'review'), ('a_status', None, 'extracted')],
    [('a', 1, 'extracted'), ('a', 2, 'extracted'), ('a_t1', None, 'extracted')],
])
def test_wide_export_rejects_column_collisions(fields):
    records = [{'paper_id': 'S1', 'field': name, 'table': table, 'status': status,
                'value': 'value'} for name, table, status in fields]
    with pytest.raises(ValueError, match='column'):
        to_wide(records)


@pytest.mark.parametrize('field', [
    {'type': 'text', 'query': ['question']},
    {'type': 'text', 'definition': {}},
    {'type': 'number', 'range': [0, float('inf')]},
    {'type': 'text', 'values': False},
    {'type': 'text', 'terms': []},
])
def test_codebook_rejects_malformed_metadata(field):
    with pytest.raises(ValueError):
        load_codebook({'field': field})


@pytest.mark.parametrize('matrix', [np.ones((1, 2)), np.full((2, 2), np.nan),
                                  np.ones((2, 3)), np.empty((2, 0))])
def test_retrieval_rejects_invalid_embeddings(matrix):
    chunks = [{'text': 'a'}, {'text': 'b'}]
    with pytest.raises(ValueError, match='embedding'):
        retrieve('q', chunks, matrix, Encoder())


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5])
def test_retrieval_rejects_invalid_passage_count(top_k):
    with pytest.raises(ValueError, match='top_k'):
        retrieve('q', [{'text': 'a'}], np.ones((1, 2)), Encoder(), top_k=top_k)


def test_retrieval_cache_tracks_document_and_encoder():
    cache = {}
    extract_with_rag('first paper', {'a': 'q'}, lambda p: '', Encoder(),
                     permission='unchecked', cache=cache)
    result = extract_with_rag('second paper', {'a': 'q'}, lambda p: '', Encoder(),
                              permission='unchecked', cache=cache)
    assert result['a']['passages'][0]['text'] == 'second paper'


def test_numeric_agreement_without_valid_pairs_has_empty_metrics():
    row = evaluate([{'paper_id': 'S1', 'field': 'n', 'value': 20}],
                   [{'paper_id': 'S1', 'field': 'n', 'value': 'invalid'}])[0]
    assert row['agreement'] == 0
    assert row['mae'] is None and row['bias'] is None and row['icc'] is None


@pytest.mark.parametrize('ratings', [[[1], [2]], [[1, np.inf], [2, 3]]])
def test_icc_rejects_invalid_ratings(ratings):
    with pytest.raises(ValueError, match='ratings'):
        icc(ratings)


def test_kappa_rejects_unpaired_labels():
    with pytest.raises(ValueError, match='length'):
        kappa(['a', 'b'], ['a'])


def build_comparison():
    records = [{'paper_id': 'S1', 'table': 1, 'outcome': 'reading', 'time': 'post',
                'arm': role, 'group': role, 'stat': stat, 'value': value,
                'status': 'extracted', 'quote': 'source'}
               for role in ('treatment', 'control')
               for stat, value in {'mean': 10, 'sd': 2, 'n': 20}.items()]
    return build_comparisons(records, studies={'S1': {'design': 'rct'}})[0]


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), 'invalid', True])
def test_effect_rejects_invalid_variance_override(value):
    found = build_comparison()
    found['inputs'].update({'var_g': value})
    result = compute_effect(found)
    assert result['status'] == 'not_computed' and result['g'] is None
    assert 'var_g_invalid' in result['checks']


@pytest.mark.parametrize('stat, value', [('mean', float('nan')), ('sd', float('inf')),
                                      ('n', 2.5), ('n', True), ('mean', 'invalid')])
def test_effect_rejects_invalid_arm_values(stat, value):
    found = build_comparison()
    found['arms']['treatment']['post'].update({stat: value})
    result = compute_effect(found)
    assert result['status'] == 'not_computed' and result['g'] is None


@pytest.mark.parametrize('stat', ['mse', 'f'])
def test_effect_rejects_negative_test_values(stat):
    found = build_comparison()
    [arm['post'].pop('sd') for arm in found['arms'].values()]
    found['tests'].update({stat: {'value': -2, 'df': [1, 38]}})
    result = compute_effect(found)
    assert result['status'] == 'not_computed' and result['g'] is None


def test_retrieval_cache_detects_page_edits():
    cache, pages, encoder = {}, [{'text': 'first paper'}], Encoder()
    extract_with_rag(pages, {'a': 'q'}, lambda p: '', encoder,
                     permission='unchecked', cache=cache)
    pages[0].update({'text': 'second paper'})
    result = extract_with_rag(pages, {'a': 'q'}, lambda p: '', encoder,
                              permission='unchecked', cache=cache)
    assert result['a']['passages'][0]['text'] == 'second paper'


def test_codebook_workbook_preserves_formula_like_text(tmp_path):
    book = {'formula': {'type': 'category', 'values': ['=1+1'],
                        'terms': {'=1+1': ['=literal']}, 'definition': '=definition'}}
    path = write_codebook_table(book, tmp_path / 'codebook.xlsx')
    assert read_codebook_table(path) == load_codebook(book)


def test_coding_workbook_preserves_formula_like_text(tmp_path):
    import openpyxl

    path = write_coding_sheet([{'paper_id': 'S1', 'field': 'text', 'value': '=1+1',
                               'quote': '=source'}], tmp_path / 'coding.xlsx')
    book = openpyxl.load_workbook(path)
    cells = list(book.active[2])
    assert cells[2].value == '=1+1' and cells[2].data_type == 's'
    assert cells[5].value == '=source' and cells[5].data_type == 's'
    book.close()


def test_empty_coding_workbook_reports_missing_columns(tmp_path):
    import openpyxl

    path = tmp_path / 'empty.xlsx'
    book = openpyxl.Workbook()
    book.save(path)
    with pytest.raises(ValueError, match='columns'):
        read_coding_sheet(path)


def test_construct_mapping_leaves_ambiguous_normalized_names_open():
    result = map_constructs(['Math score'], ['math_score', 'math-score'])
    assert result['mapping'] == {} and result['unmapped'] == ['Math score']


def test_matrix_export_rejects_asymmetry(tmp_path):
    study = {'study': 'S1', 'labels': ['a', 'b'], 'matrix': [[1, .2], [.3, 1]],
             'n': 20, 'mapping': {'a': 'a', 'b': 'b'}}
    with pytest.raises(ValueError, match='symmetric'):
        to_matrices([study], ['a', 'b'], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('value', [float('inf'), float('-inf'), 10 ** 400])
def test_model_number_must_be_finite(value):
    import json

    book = load_codebook({'number': {'type': 'number'}})
    records = extract_values([{'text': 'The source reports a measured value.'}], book,
                             respond=lambda p: json.dumps({'value': value, 'passage': 1,
                                                           'quote': 'reports a measured value'}),
                             encoder=Encoder(), permission=check_llm_use(
                                 access='open_access', license_type='cc-by', processing='api'))
    assert records[0]['status'] == 'review' and records[0]['value'] is None


@pytest.mark.parametrize('passage', [None, 0, True, 99])
def test_model_without_valid_citation_needs_review(passage):
    import json

    book = load_codebook({'text': {'type': 'text'}})
    records = extract_values([{'text': 'The source reports a measured value.'}], book,
                             respond=lambda p: json.dumps({'value': 'measured value',
                                                           'passage': passage,
                                                           'quote': 'reports a measured value'}),
                             encoder=Encoder(), permission=check_llm_use(
                                 access='open_access', license_type='cc-by', processing='api'))
    assert records[0]['status'] == 'review'


@pytest.mark.parametrize('content', [42, [], {}, ''])
def test_provider_rejects_nontext_reply(content):
    respond, _ = make_responder('ollama', 'model', post=lambda *args: {
        'choices': [{'message': {'content': content}}]})
    with pytest.raises(RuntimeError, match='reply text'):
        respond('prompt')


def test_retrieval_marks_nontext_reply_as_model_failure():
    result = extract_with_rag('source text', {'a': 'q'}, lambda p: {}, Encoder(),
                              permission='unchecked')['a']
    assert result['error'] == 'ValueError' and result['escalate']


def test_codebook_rejects_integer_outside_float_range():
    with pytest.raises(ValueError, match='range'):
        load_codebook({'number': {'type': 'number', 'range': [0, 10 ** 400]}})


def test_effect_rejects_integer_outside_float_range():
    found = build_comparison()
    found['arms']['treatment']['post'].update({'mean': 10 ** 400})
    assert compute_effect(found)['status'] == 'not_computed'


@pytest.mark.parametrize('constructs', [['construct', 'b'], ['a', 'a']])
def test_matrix_export_rejects_reserved_or_duplicate_columns(tmp_path, constructs):
    study = {'study': 'S1', 'labels': ['a', 'b'], 'matrix': np.eye(2), 'n': 20,
             'mapping': {'a': constructs[0], 'b': constructs[1]}}
    with pytest.raises(ValueError, match='construct'):
        to_matrices([study], constructs, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
