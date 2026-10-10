import pytest

from pubextractor import (
    AGREEMENT_COLUMNS,
    evaluate,
    graph_agreement,
    icc,
    kappa,
    read_coding_sheet,
    write_coding_sheet,
)


def pairs(field, extracted, coded):
    left = [{'paper_id': f"S{i}", 'field': field, 'value': v, 'status': 'extracted'}
            for i, v in enumerate(extracted)]
    right = [{'paper_id': f"S{i}", 'field': field, 'value': v} for i, v in enumerate(coded)]
    return left, right


def test_kappa_matches_hand_worked_value():
    # hand-worked: po = 4/6, pe = 0.5, kappa = (4/6 - 0.5) / 0.5 = 1/3
    assert kappa(list('xxyyxy'), list('xyyyxx')) == pytest.approx(1 / 3)


def test_kappa_matches_sklearn():
    metrics = pytest.importorskip('sklearn.metrics')
    a, b = list('aabbcabcca'), list('abbbcacbca')
    assert kappa(a, b) == pytest.approx(metrics.cohen_kappa_score(a, b))


def test_kappa_is_none_when_chance_agreement_is_one():
    assert kappa(['x', 'x'], ['x', 'x']) is None


def test_icc_matches_mean_squares_by_hand():
    # hand-worked mean squares: MSR = 91/24, MSC = 9/8, MSE = 1/8, ICC(2,1) = 44/53
    assert icc([[1, 2], [2, 2], [3, 4], [4, 5]]) == pytest.approx(44 / 53)


def test_icc_matches_shrout_fleiss_table_2():
    # Shrout and Fleiss (1979), Table 2 data and Table 4 ICC(2,1) = .29
    ratings = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9],
               [6, 2, 4, 7]]
    assert icc(ratings) == pytest.approx(0.29, abs=0.005)


def test_evaluate_numeric_field():
    # hand-worked: differences -1, 0, -1, -1; bias -0.75, SD 0.5, limits -0.75 +/- 0.98
    rows = evaluate(*pairs('n', [1, 2, 3, 4], [2, 2, 4, 5]))
    assert list(rows[0]) == list(AGREEMENT_COLUMNS)
    row = rows[0]
    assert row['n_pairs'] == 4
    assert row['agreement'] == pytest.approx(25.0)
    assert row['kappa'] is None
    assert row['icc'] == pytest.approx(44 / 53)
    assert row['mae'] == pytest.approx(0.75)
    assert row['bias'] == pytest.approx(-0.75)
    assert row['lower'] == pytest.approx(-1.73)
    assert row['upper'] == pytest.approx(0.23)


def test_evaluate_categorical_field():
    row = evaluate(*pairs('design', list('xxyyxy'), list('xyyyxx')))[0]
    assert row['agreement'] == pytest.approx(400 / 6)
    assert row['kappa'] == pytest.approx(1 / 3)
    assert (row['icc'], row['mae'], row['bias']) == (None, None, None)


def test_hallucinations_and_omissions():
    extracted, coded = pairs('g', ['a', 'b', None, 'c', 'd'], ['a', 'not_found', 'b', '', 'd'])
    row = evaluate(extracted, coded)[0]
    assert (row['hallucinations'], row['omissions']) == (2, 1)
    assert row['agreement'] == pytest.approx(40.0)


def test_status_not_found_counts_as_no_value():
    extracted, coded = pairs('g', ['a'], ['a'])
    extracted[0].update({'status': 'not_found'})
    assert evaluate(extracted, coded)[0]['omissions'] == 1


def test_numeric_tolerance():
    assert evaluate(*pairs('m', [10.004], [10.0]))[0]['agreement'] == 100.0
    assert evaluate(*pairs('m', [10.4], [10.0]))[0]['agreement'] == 0.0
    assert evaluate(*pairs('m', [10.4], [10.0]), tolerance=0.5)[0]['agreement'] == 100.0


def test_only_shared_keys_are_paired():
    extracted, coded = pairs('m', [1, 2, 3], [1, 2])
    assert evaluate(extracted, coded)[0]['n_pairs'] == 2


def test_graph_agreement():
    manual = [{'case': 'A', 'session': s, 'outcome': y} for s, y in ((1, 10), (2, 20), (3, 30))]
    extracted = [{'case': 'A', 'session': s, 'outcome': y} for s, y in ((1, 11), (2, 18), (4, 40))]
    result = graph_agreement(extracted, manual, y_range=100)
    assert result == {'n_pairs': 2, 'mae': pytest.approx(1.5), 'mae_percent': pytest.approx(1.5),
                      'missing': 1, 'extra': 1}
    assert graph_agreement(extracted, manual)['mae_percent'] == pytest.approx(7.5)


def test_graph_agreement_reads_csv_strings():
    manual = [{'case': 'A', 'session': '1.02', 'outcome': '10'}]
    extracted = [{'case': 'A', 'session': '1', 'outcome': '12.5'}]
    assert graph_agreement(extracted, manual, y_range=50)['mae_percent'] == pytest.approx(5.0)


def test_coding_sheet_round_trip(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')
    records = [{'paper_id': 'S01', 'field': 'design', 'value': 'group', 'status': 'extracted'},
               {'paper_id': 'S01', 'field': 'sample_size', 'value': 46, 'status': 'review'}]
    path = write_coding_sheet(records, tmp_path / 'coding.xlsx')
    book = openpyxl.load_workbook(path)
    sheet = book.active
    header = [cell.value for cell in sheet[1]]
    assert header[-1] == 'coded_value'
    assert [cell.value for cell in sheet['B'][1:]] == ['design', 'sample_size']
    assert sheet.cell(2, len(header)).value is None
    sheet.cell(2, len(header)).value = 'single_case'
    sheet.cell(3, len(header)).value = 46
    book.save(path)
    coded = read_coding_sheet(path)
    assert [(r['field'], r['value']) for r in coded] == [('design', 'single_case'),
                                                         ('sample_size', 46)]
    assert 'coded_value' not in coded[0]
    rows = {row['field']: row for row in evaluate(records, coded)}
    assert rows['design']['agreement'] == 0.0
    assert rows['sample_size']['agreement'] == 100.0


def test_kappa_is_blank_with_one_pair():
    from pubextractor.evaluate import kappa
    assert kappa(['group'], ['single_case']) is None


def test_text_typed_into_a_number_field_counts_as_disagreement():
    from pubextractor.evaluate import evaluate
    extracted = [{'paper_id': p, 'field': 'n', 'value': v} for p, v in (('a', 46), ('b', 60), ('c', 30))]
    coded = [{'paper_id': 'a', 'field': 'n', 'value': 'forty'}, {'paper_id': 'b', 'field': 'n', 'value': 60},
             {'paper_id': 'c', 'field': 'n', 'value': 31}]
    row = evaluate(extracted, coded)[0]
    assert row['kappa'] is None and row['mae'] == 0.5
    assert round(row['agreement'], 1) == 33.3
