# tests/test_cells.py
import pytest

from pubextractor.cells import header_n, parse_cell


class TestParseCell:
    @pytest.mark.parametrize('text, kind, value, second, markers', [
        ('12.3', 'number', 12.3, None, []),
        ('−0.31', 'number', -0.31, None, []),
        ('1,024', 'number', 1024.0, None, []),
        ('.45', 'number', 0.45, None, []),
        ('12.3a', 'number', 12.3, None, ['a']),
        ('0.62**', 'number', 0.62, None, ['**']),
        ('12.3 (4.5)', 'mean_sd', 12.3, 4.5, []),
        ('12.3a (4.5)', 'mean_sd', 12.3, 4.5, ['a']),
        ('12.3 ± 4.5', 'mean_sd', 12.3, 4.5, []),
        ('45%', 'percent', 45.0, None, []),
        ('12 (40.0%)', 'n_percent', 12.0, 40.0, []),
    ])
    def test_numbers(self, text, kind, value, second, markers):
        assert parse_cell(text) == {'kind': kind, 'value': value, 'second': second,
                                    'markers': markers}

    @pytest.mark.parametrize('text', ['—', '–', '-', '--', 'n/a', 'NA', 'n.r.', 'NR',
                                      'N.R.', 'not reported'])
    def test_dashes_mean_missing(self, text):
        assert parse_cell(text)['kind'] == 'missing' and parse_cell(text)['value'] is None

    @pytest.mark.parametrize('text, kind', [
        ('', 'empty'), (None, 'empty'), ('  ', 'empty'),
        ('Treatment', 'text'), ('12,3', 'text'), ('12–15', 'text'), ('12 ms', 'text'),
    ])
    def test_other_cells(self, text, kind):
        assert parse_cell(text)['kind'] == kind


class TestHeaderN:
    @pytest.mark.parametrize('text, expected', [
        ('Treatment (n = 24)', ('Treatment', 24)),
        ('Control, N=1,024', ('Control', 1024)),
        ('Treatment(n=24) M', ('Treatment M', 24)),
        ('Mean', ('Mean', None)),
        ('', ('', None)),
    ])
    def test_label_and_n(self, text, expected):
        assert header_n(text) == expected
