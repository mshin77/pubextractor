import json

import numpy as np
import pytest

from pubextractor.confidence import (
    confidence_report,
    needs_escalation,
    score_answer,
    score_extractions,
)
from pubextractor.provenance import ProvenanceTracker, compare_provenances, log_human
from pubextractor.rag import chunk_pages, chunk_text, extract_with_rag


class Encoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 2))


PAGES = [{'page': 1, 'text': 'some text here'}]


def test_unchecked_permission_is_recorded():
    tracker = ProvenanceTracker('s', 't')
    extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(), tracker=tracker,
                     permission='unchecked', paper_id='P1')
    assert tracker.to_dict()['permissions']['P1']['verdict'] == 'unchecked'


@pytest.mark.parametrize('answer', [
    '"entirely invented quote here" passage 1 p. 5 42',
    '\u201centirely invented quote here\u201d passage 1 p. 5 42',
])
def test_invented_quote_escalates(answer):
    score = score_answer(answer, ['unrelated text'])
    assert score < 0.7
    assert needs_escalation(score)


def test_hyphen_line_break_quote_scores_high():
    passage = 'the single-\ncase design was used in all participants here'
    answer = '"single-case design was used in all" passage 1 p. 5'
    assert score_answer(answer, [passage]) >= 0.7


def test_removed_hyphen_quote_scores_high():
    passage = 'the multi-\nple baseline design was used in all participants'
    answer = '"multiple baseline design was used" passage 1 p. 5'
    assert score_answer(answer, [passage]) >= 0.7


def test_curly_quote_and_dash_genuine_scores_high():
    passage = "the teacher\u2019s rating \u2013 measured weekly \u2013 rose sharply here"
    answer = "\u201cthe teacher's rating - measured weekly - rose\u201d passage 1 p. 5"
    assert score_answer(answer, [passage]) >= 0.7


def test_ellipsis_quote_genuine_scores_high():
    passage = 'the design used a multiple baseline and then later the effect was large across participants'
    answer = '"the design used a multiple baseline ... the effect was large across" passage 1 p. 5'
    assert score_answer(answer, [passage]) >= 0.7


def test_ellipsis_quote_with_invented_segment_escalates():
    passage = 'the design used a multiple baseline and then later the effect was large across participants'
    answer = '"the design used a multiple baseline \u2026 entirely invented segment here" passage 1 p. 5'
    assert needs_escalation(score_answer(answer, [passage]))


def test_found_quote_scores_higher_than_invented():
    passage = 'The   Design used a MULTIPLE baseline across participants.'
    quoted = 'The design used a multiple baseline across'
    found = score_answer(f'"{quoted}" passage 1 p. 5 42', [passage])
    invented = score_answer('"entirely invented quote here" passage 1 p. 5 42', [passage])
    assert found > invented


def test_query_parameter_removed():
    with pytest.raises(TypeError):
        score_answer('x', [], 'query')


def test_score_extractions_length_mismatch():
    with pytest.raises(ValueError, match='length'):
        score_extractions([{'answer': 'a'}, {'answer': 'b'}], [['p']])


def test_report_recomputes_escalate_with_own_threshold():
    scored = [{'confidence': 0.75, 'confidence_band': 'moderate', 'escalate': False}]
    assert confidence_report(scored, threshold=0.9)['escalate'] == 1


@pytest.mark.parametrize('overlap', [-1, 10, 11])
def test_chunk_text_rejects_bad_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_text('word ' * 100, chunk_size=10, overlap=overlap)


def test_chunk_pages_rejects_non_dict_pages():
    with pytest.raises(ValueError, match='page'):
        chunk_pages(['a', 'b'])


def test_extract_rejects_none_text():
    with pytest.raises(ValueError, match='text'):
        extract_with_rag(None, {'a': 'q'}, lambda p: '', Encoder(), permission='unchecked')


def test_extract_rejects_malformed_query():
    with pytest.raises(ValueError, match='query'):
        extract_with_rag(PAGES, {'a': {'x': 1}}, lambda p: '', Encoder(), permission='unchecked')


def test_paper_id_zero_kept():
    tracker = ProvenanceTracker('s', 't').log_permission({'verdict': 'v'}, paper_id=0)
    assert 0 in tracker.to_dict()['permissions'] or '0' in tracker.to_dict()['permissions']
    assert 'unnamed' not in tracker.to_dict()['permissions']


def test_prompt_model_recorded_per_prompt():
    tracker = ProvenanceTracker('s', 't').log_prompt('a', model='m1').log_prompt('b', model='m2')
    assert [r['model'] for r in tracker.to_dict()['prompts']['records']] == ['m1', 'm2']
    assert 'model' not in tracker.to_dict()['prompts']


def test_compare_ignores_record_time():
    def run(stamp):
        tracker = ProvenanceTracker('s', 't')
        log_human(tracker, 'p', 'f', 'confirm', 1, 1, coder='c', time=stamp)
        data = json.loads(json.dumps(tracker.to_dict()))
        data['created_at'] = data['environment'] = None
        return data
    assert compare_provenances(run('a'), run('b'))['identical']
