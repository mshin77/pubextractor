import pytest

from pubextractor.statistics import find_statistics


def stats(text):
    return [(s['stat'], s['value'], s['df']) for s in find_statistics(text)]


class TestFindStatistics:
    def test_descriptives(self):
        assert stats('Treatment (n = 25, M = 61.3, SD = 14.2, SE = 2.84)') == [
            ('n', 25.0, []), ('mean', 61.3, []), ('sd', 14.2, []), ('se', 2.84, [])]

    def test_total_n_with_comma(self):
        assert stats('In total, N = 1,412 students took part.') == [('n', 1412.0, [])]

    def test_trailing_comma_is_not_part_of_the_number(self):
        found = find_statistics('Control (n = 25, M = 55.0)')
        assert found[0]['text'] == 'n = 25' and found[0]['value'] == 25.0

    def test_t_and_f_with_degrees_of_freedom(self):
        assert stats('t(48) = 2.43, p = .019; F(1, 96) = 5.90') == [
            ('t', 2.43, [48.0]), ('p', 0.019, []), ('f', 5.9, [1.0, 96.0])]

    def test_correlation_with_and_without_df(self):
        assert stats('r(410) = .62 and r = \u2212.31') == [('r', 0.62, [410.0]), ('r', -0.31, [])]

    def test_p_relation(self):
        found = find_statistics('p < .001')
        assert found[0]['relation'] == '<' and found[0]['value'] == 0.001

    def test_mse(self):
        assert stats('MSE = 12.75') == [('mse', 12.75, [])]

    def test_positions_point_at_the_original_text(self):
        text = 'Scores: M = 61.3 overall.'
        found = find_statistics(text)[0]
        assert text[found['start']:found['end']] == found['text'] == 'M = 61.3'

    @pytest.mark.parametrize('text', ['mean scores rose', 'in = 3', 'SMD = 0.4'])
    def test_no_false_matches(self, text):
        assert find_statistics(text) == []


class TestBoundaries:
    @pytest.mark.parametrize('value', [None, 12, 3.5, b'n = 5', ['n = 5']])
    def test_non_string_raises_value_error(self, value):
        with pytest.raises(ValueError):
            find_statistics(value)

    def test_empty_string(self):
        assert find_statistics('') == []

    @pytest.mark.parametrize('text', ['r = \u2212.32', 'r = \u2013.32', 'r = -.32'])
    def test_minus_variants(self, text):
        assert stats(text) == [('r', -0.32, [])]

    def test_unicode_minus_keeps_original_slice(self):
        text = 'r(98) = \u2212.32'
        found = find_statistics(text)[0]
        assert found['text'] == text and found['df'] == [98.0]

    def test_no_spaces_around_equals(self):
        assert stats('SD=12.3') == [('sd', 12.3, [])]

    @pytest.mark.parametrize('text, relation, value', [
        ('p < .001', '<', 0.001), ('p<.05', '<', 0.05), ('p > .10', '>', 0.10),
        ('p \u2264 .05', '\u2264', 0.05), ('p \u2265 .05', '\u2265', 0.05),
        ('p = .049.', '=', 0.049)])
    def test_p_relations(self, text, relation, value):
        found = find_statistics(text)
        assert [(s['relation'], s['value']) for s in found] == [(relation, value)]

    def test_t_with_spaces(self):
        assert stats('t(28) = 2.45') == [('t', 2.45, [28.0])]

    def test_f_with_fractional_df(self):
        assert stats('F(1, 57.8) = 4.10') == [('f', 4.1, [1.0, 57.8])]

    def test_apa_descriptives(self):
        assert stats('M = 55.0, SD = 9.1') == [('mean', 55.0, []), ('sd', 9.1, [])]

    def test_sentence_final_comma_after_n(self):
        assert stats('Groups were equal, n = 25, and balanced.') == [('n', 25.0, [])]

    def test_relation_is_equals_outside_p(self):
        assert {s['relation'] for s in find_statistics('n = 5, SD = 1.2, t(9) = 2.0')} == {'='}

    def test_mse_not_also_read_as_se(self):
        assert [s['stat'] for s in find_statistics('MSE = 12.75')] == ['mse']

    def test_results_sorted_by_position(self):
        starts = [s['start'] for s in find_statistics('p = .02, t(9) = 2.1, n = 10, SD = 3')]
        assert starts == sorted(starts)


class TestDecimalComma:
    def test_decimal_comma_with_two_digits_is_a_decimal(self):
        found = find_statistics('t(78) = 1,79, p = .08')
        assert (found[0]['stat'], found[0]['value'], found[0]['text']) == ('t', 1.79, 't(78) = 1,79')

    def test_three_digits_after_the_comma_stay_thousands(self):
        assert stats('M = 1,234 and SD = 12,50') == [('mean', 1234.0, []), ('sd', 12.5, [])]

    def test_a_count_never_reads_a_comma_as_decimal(self):
        assert stats('n = 25,30') == [('n', 25.0, [])]


def test_decimal_comma_with_one_digit_is_read_whole():
    found = find_statistics('M = 12,5, SD = 3,1')
    assert [(item['stat'], item['value']) for item in found] == [('mean', 12.5), ('sd', 3.1)]
