import numpy as np
import pytest

from pubextractor.licensing import check_llm_use
from pubextractor.provenance import ProvenanceTracker, compare_provenances, load_provenance
from pubextractor.rag import build_index, chunk_pages, chunk_text, extract_with_rag, retrieve


class StubEncoder:
    """Encodes on keyword overlap, so retrieval order is predictable in tests."""

    vocabulary = ('dyslexia', 'university', 'fraction', 'baseline')

    def encode(self, texts, batch_size=None, show_progress_bar=None):
        return np.array([[float(word in text.lower()) for word in self.vocabulary]
                         for text in texts])


CHUNKS = [
    'Students with dyslexia at university reported unmet needs.',
    'A multiple baseline design across four participants was used.',
    'Fraction magnitude was estimated on a number line.',
]


class TestChunking:
    def test_overlap_and_coverage(self):
        chunks = chunk_text('word ' * 400, chunk_size=500, overlap=100)
        assert len(chunks) > 1
        assert chunks[1]['start'] < chunks[0]['start'] + 500

    def test_rejects_overlap_at_or_above_size(self):
        with pytest.raises(ValueError):
            chunk_text('text', chunk_size=100, overlap=100)

    def test_short_text_is_one_chunk(self):
        assert len(chunk_text('short text', chunk_size=500, overlap=50)) == 1

    def test_indices_are_sequential(self):
        chunks = chunk_text('word ' * 300, chunk_size=400, overlap=80)
        assert [c['index'] for c in chunks] == list(range(len(chunks)))


class TestRetrieval:
    def test_ranks_relevant_chunk_first(self):
        model = StubEncoder()
        chunks = [{'index': i, 'start': 0, 'text': t} for i, t in enumerate(CHUNKS)]
        embeddings = build_index(chunks, model)
        top = retrieve('dyslexia at university', chunks, embeddings, model, top_k=2)
        assert top[0]['index'] == 0
        assert top[0]['similarity'] >= top[1]['similarity']

    def test_top_k_limits_results(self):
        model = StubEncoder()
        chunks = [{'index': i, 'start': 0, 'text': t} for i, t in enumerate(CHUNKS)]
        embeddings = build_index(chunks, model)
        assert len(retrieve('fraction', chunks, embeddings, model, top_k=1)) == 1

    def test_zero_vector_query_does_not_divide_by_zero(self):
        model = StubEncoder()
        chunks = [{'index': 0, 'start': 0, 'text': 'unrelated words entirely'}]
        embeddings = build_index(chunks, model)
        results = retrieve('nothing matching', chunks, embeddings, model)
        assert np.isfinite(results[0]['similarity'])


class TestExtraction:
    def test_grounded_answer_is_not_escalated(self):
        text = ' '.join(CHUNKS)
        def responder(prompt):
            return ('The study states "Students with dyslexia at '
                    'university reported unmet needs" in passage 1, p. 3')
        results = extract_with_rag(text, {'population': 'who participated'}, responder,
                                   StubEncoder(), chunk_size=200, overlap=40, permission='unchecked')
        assert results['population']['escalate'] is False
        assert results['population']['passages']

    def test_not_found_is_escalated(self):
        results = extract_with_rag(' '.join(CHUNKS), {'design': 'what design'},
                                   lambda p: 'not found', StubEncoder(),
                                   chunk_size=200, overlap=40, permission='unchecked')
        assert results['design']['escalate'] is True
        assert results['design']['confidence_band'] == 'very low'

    def test_provider_failure_is_contained_and_escalated(self):
        def raising(prompt):
            raise RuntimeError('provider down')

        results = extract_with_rag(' '.join(CHUNKS), {'design': 'what design'},
                                   raising, StubEncoder(), chunk_size=200, overlap=40, permission='unchecked')
        assert results['design']['answer'].startswith('ERROR')
        assert results['design']['escalate'] is True

    def test_records_prompts_and_retrieval_in_provenance(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        extract_with_rag(' '.join(CHUNKS), {'a': 'q1', 'b': 'q2'},
                         lambda p: 'not found', StubEncoder(),
                         chunk_size=200, overlap=40, tracker=tracker, permission='unchecked')
        recorded = tracker.to_dict()
        assert recorded['prompts']['count'] == 2
        assert recorded['prompts']['digest']
        assert recorded['retrieval']['chunk_size'] == 200
        assert recorded['retrieval']['top_k'] == 5


PAGES = [
    {'page': 1, 'printed_page': 576, 'section': 'Method', 'text': CHUNKS[1]},
    {'page': 2, 'printed_page': None, 'section': 'Results', 'text': CHUNKS[2]},
]


class TestPageAnchors:
    def test_chunks_carry_page_and_section(self):
        chunks = chunk_pages(PAGES, chunk_size=200, overlap=40)
        assert [c['pdf_page'] for c in chunks] == [1, 2]
        assert chunks[0]['printed_page'] == 576
        assert chunks[1]['section'] == 'Results'
        assert [c['index'] for c in chunks] == [0, 1]

    def test_chunks_never_cross_pages(self):
        pages = [{'page': 1, 'text': 'word ' * 300}, {'page': 2, 'text': 'word ' * 300}]
        chunks = chunk_pages(pages, chunk_size=400, overlap=80)
        assert {c['pdf_page'] for c in chunks} == {1, 2}
        assert all(len(c['text']) <= 400 for c in chunks)

    def test_prompt_and_passages_show_page_anchor(self):
        prompts = []
        results = extract_with_rag(PAGES, {'design': 'multiple baseline'},
                                   lambda p: prompts.append(p) or 'not found',
                                   StubEncoder(), chunk_size=200, overlap=40, top_k=2, permission='unchecked')
        assert 'p. 576 · Method' in prompts[0]
        assert 'file p. 2 · Results' in prompts[0]
        assert results['design']['passages'][0]['anchor'] == 'p. 576 · Method'

    def test_plain_text_keeps_chunk_labels(self):
        prompts = []
        extract_with_rag(' '.join(CHUNKS), {'a': 'fraction'},
                         lambda p: prompts.append(p) or 'not found',
                         StubEncoder(), chunk_size=200, overlap=40, permission='unchecked')
        assert '(chunk 0)' in prompts[0]


class TestPermission:
    def test_refused_permission_sends_nothing(self):
        permission = check_llm_use(access='subscription', ai_terms='silent', processing='api')
        sent = []
        with pytest.raises(PermissionError, match='silent'):
            extract_with_rag(PAGES, {'a': 'q'}, lambda p: sent.append(p) or '',
                             StubEncoder(), permission=permission)
        assert sent == []

    def test_granted_permission_is_recorded(self):
        permission = check_llm_use(access='subscription', ai_terms='silent', processing='local')
        tracker = ProvenanceTracker('fulltext', 'extraction')
        extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', StubEncoder(),
                         chunk_size=200, overlap=40, tracker=tracker, permission=permission,
                         endpoint='http://localhost:11434/v1')
        assert tracker.to_dict()['permissions']['unnamed']['verdict'] == 'local_only'


class TestProvenance:
    def test_environment_and_version_recorded(self):
        recorded = ProvenanceTracker('screening', 'titles').to_dict()
        assert recorded['environment']['pubextractor']
        assert recorded['environment']['python']

    def test_criteria_recorded_verbatim(self):
        criteria = {'dyslexia': 'The record addresses dyslexia.'}
        recorded = ProvenanceTracker('s', 't').log_criteria(criteria).to_dict()
        assert recorded['criteria'] == criteria

    def test_prompt_digest_changes_with_content(self):
        first = ProvenanceTracker('s', 't').log_prompt('one').to_dict()['prompts']['digest']
        second = ProvenanceTracker('s', 't').log_prompt('two').to_dict()['prompts']['digest']
        assert first != second

    def test_save_and_load_roundtrip(self, tmp_path):
        tracker = ProvenanceTracker('s', 't').log_random_seed(2026).log_threshold(0.31)
        path = tracker.save(tmp_path / 'prov.json')
        loaded = load_provenance(path)
        assert loaded['random_seed'] == 2026
        assert loaded['threshold']['value'] == 0.31

    def test_compare_ignores_timestamp(self):
        left = ProvenanceTracker('s', 't').log_random_seed(1).to_dict()
        right = ProvenanceTracker('s', 't').log_random_seed(1).to_dict()
        assert compare_provenances(left, right)['identical'] is True

    def test_compare_reports_differing_field(self):
        left = ProvenanceTracker('s', 't').log_threshold(0.3).to_dict()
        right = ProvenanceTracker('s', 't').log_threshold(0.5).to_dict()
        result = compare_provenances(left, right)
        assert result['identical'] is False
        assert 'threshold.value' in result['differences']


class TestChunkingAlwaysMovesForward:
    def test_a_long_unbroken_run_does_not_stall(self):
        from pubextractor.rag import chunk_text

        text = 'x ' + 'a' * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3
        assert [c['start'] for c in chunks] == sorted({c['start'] for c in chunks})

    def test_every_chunk_starts_after_the_one_before(self):
        from pubextractor.rag import chunk_text

        text = ' '.join('word' for _ in range(600))
        starts = [c['start'] for c in chunk_text(text, chunk_size=400, overlap=120)]
        assert all(b > a for a, b in zip(starts, starts[1:]))


def test_chunks_without_page_key_have_no_pdf_page_and_keep_block_numbers():
    chunks = chunk_pages([{'text': 'first block', 'paragraph': 4, 'section': 'Method'}])
    assert (chunks[0]['pdf_page'], chunks[0]['paragraph'], chunks[0]['table']) == (None, 4, None)
