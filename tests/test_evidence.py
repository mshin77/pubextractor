
from pubextractor.confidence import (
    confidence_report,
    interpret_confidence,
    needs_escalation,
    score_answer,
    score_extractions,
)
from pubextractor.evidence import (
    find_keyword_spans,
    format_evidence,
    highlight_markdown,
    search_document,
)

TEXT = 'Students with dyslexia at university reported unmet needs in assessment'


class TestKeywordSpans:
    def test_offsets_locate_the_term(self):
        span = find_keyword_spans(TEXT, 'dyslexia', window=3)[0]
        assert span['text'][span['match_start']:span['match_end']] == 'dyslexia'

    def test_asymmetric_window(self):
        span = find_keyword_spans(TEXT, 'university', window=(2, 0))[0]
        assert span['text'].endswith('university')

    def test_case_insensitive_by_default(self):
        assert find_keyword_spans('Dyslexia matters', 'dyslexia')
        assert not find_keyword_spans('Dyslexia matters', 'dyslexia', case_sensitive=True)

    def test_whole_word_only(self):
        assert not find_keyword_spans('predyslexial notions', 'dyslexia')

    def test_max_matches(self):
        text = 'dyslexia and dyslexia and dyslexia'
        assert len(find_keyword_spans(text, 'dyslexia', max_matches=2)) == 2

    def test_literal_not_pattern(self):
        assert find_keyword_spans('a c++ course', 'c++')

    def test_context_bounds_are_absolute(self):
        span = find_keyword_spans(TEXT, 'dyslexia', window=3)[0]
        assert TEXT[span['context_start']:span['context_end']] == span['text']

    def test_close_hits_report_overlapping_context(self):
        text = 'dyslexia and dyslexia again'
        first, second = find_keyword_spans(text, 'dyslexia', window=5)
        assert first['context_start'] < second['context_end']
        assert second['context_start'] < first['context_end']

    def test_distant_hits_report_separate_context(self):
        filler = ' '.join(['filler'] * 40)
        text = f'dyslexia {filler} dyslexia'
        first, second = find_keyword_spans(text, 'dyslexia', window=3)
        assert first['context_end'] < second['context_start']


class TestSearchDocument:
    def test_anchors_on_printed_page(self):
        pages = [{'page': 2, 'printed_page': 188, 'section': 'Results',
                  'text': 'Dyslexia at university.'}]
        span = search_document(pages, ['university'])['university'][0]
        assert span['anchor'] == 'p. 188 · Results'
        assert span['printed_page'] == 188
        assert span['pdf_page'] == 2

    def test_marks_file_position_when_printed_unknown(self):
        pages = [{'page': 2, 'printed_page': None, 'section': '', 'text': 'university'}]
        assert search_document(pages, ['university'])['university'][0]['anchor'] == 'file p. 2'

    def test_absent_keyword_returns_empty_list(self):
        pages = [{'page': 1, 'printed_page': 1, 'section': '', 'text': 'nothing here'}]
        assert search_document(pages, ['university'])['university'] == []


class TestRendering:
    def test_markdown_bolds_only_the_match(self):
        span = find_keyword_spans(TEXT, 'dyslexia', window=2)[0]
        assert '**dyslexia**' in highlight_markdown(span)
        assert highlight_markdown(span).count('**') == 2

    def test_format_evidence_respects_limit(self):
        pages = [{'page': 1, 'printed_page': 1, 'section': '',
                  'text': 'dyslexia one. dyslexia two. dyslexia three.'}]
        spans = search_document(pages, ['dyslexia'])['dyslexia']
        assert format_evidence(spans, max_spans=2).count('|') == 1

    def test_format_evidence_empty(self):
        assert format_evidence([]) == ''


class TestConfidence:
    def test_hedging_lowers_score(self):
        assert score_answer('It is unclear whether the study reports this') < 0.5

    def test_error_is_zero(self):
        assert score_answer('ERROR: call failed') == 0.0

    def test_empty_is_zero(self):
        assert score_answer('') == 0.0
        assert score_answer(None) == 0.0

    def test_quote_and_page_reference_raise_score(self):
        grounded = score_answer('The authors state "a multiple baseline design '
                                'across four participants" on p. 12')
        assert grounded > 0.7

    def test_passage_overlap_raises_score(self):
        answer = 'participants were four sixth grade students with mathematics difficulties'
        with_passages = score_answer(answer, [answer])
        assert with_passages > score_answer(answer)

    def test_bands(self):
        assert interpret_confidence(0.9) == 'high'
        assert interpret_confidence(0.65) == 'moderate'
        assert interpret_confidence(0.45) == 'low'
        assert interpret_confidence(0.1) == 'very low'
        assert interpret_confidence(None) is None

    def test_escalation_on_low_and_missing(self):
        assert needs_escalation(0.4) is True
        assert needs_escalation(0.9) is False
        assert needs_escalation(None) is True

    def test_score_extractions_flags(self):
        scored = score_extractions([{'answer': 'not found'},
                                    {'answer': 'The study states "four participants" p. 3'}])
        assert scored[0]['escalate'] is True
        assert scored[1]['escalate'] is False
        assert scored[0]['confidence_band'] == 'very low'

    def test_report_counts(self):
        scored = score_extractions([{'answer': 'not found'}, {'answer': 'unclear'}])
        report = confidence_report(scored)
        assert report['total'] == 2
        assert report['escalate'] == 2
        assert sum(report['bands'].values()) == 2

    def test_report_empty(self):
        assert confidence_report([])['total'] == 0
