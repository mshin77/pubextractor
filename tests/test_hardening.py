import logging

import numpy as np
import pytest

from pubextractor.fulltext import similarity_scores
from pubextractor.licensing import check_llm_use, is_local_endpoint
from pubextractor.provenance import ProvenanceTracker
from pubextractor.rag import chunk_pages, extract_with_rag

PAGES = [{'page': 1, 'text': 'A multiple baseline design across four participants.'}]


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[float('baseline' in t), 1.0] for t in texts])


def run(**kwargs):
    return extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(), **kwargs)


class TestEndpointParsing:
    @pytest.mark.parametrize('url', [
        'http://evil.com\\@localhost:11434',
        'http://evil.com%00.localhost',
        'http://user@localhost:11434',
        'http://evil.localhost',
        'http://localhost.evil.com',
        'ftp://localhost',
        'localhost:11434',
        'http://localhost:99999',
        b'http://localhost',
        123,
    ])
    def test_rejected(self, url):
        assert not is_local_endpoint(url)

    @pytest.mark.parametrize('url', ['http://localhost:11434/v1', 'https://127.0.0.2:8000',
                                     'http://[::1]:11434', 'http://LOCALHOST.:11434',
                                     'http://[::ffff:127.0.0.1]:80'])
    def test_accepted(self, url):
        assert is_local_endpoint(url)


class TestPermissionIntegrity:
    def test_forged_permission_is_refused(self):
        forged = {'verdict': 'manual', 'allowed': True, 'reason': 'x', 'inputs': {}}
        with pytest.raises(ValueError, match='permission'):
            run(permission=forged, endpoint='https://api.openai.com')

    def test_relabeled_processing_is_refused(self):
        permission = check_llm_use(access='subscription', ai_terms='silent', processing='local')
        permission['inputs'] = {**permission['inputs'], 'processing': 'api'}
        with pytest.raises(ValueError, match='does not match'):
            run(permission=permission)

    def test_string_allowed_is_refused(self):
        permission = check_llm_use(access='subscription', ai_terms='silent', processing='api')
        permission['allowed'] = 'False'
        with pytest.raises(ValueError, match='does not match'):
            run(permission=permission, endpoint='https://api.openai.com')

    def test_missing_permission_is_refused(self):
        with pytest.raises(ValueError, match='unchecked'):
            run()

    def test_explicit_unchecked_runs_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='pubextractor.rag'):
            assert 'a' in run(permission='unchecked')
        assert 'unchecked' in caplog.text

    def test_api_endpoint_is_recorded(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        run(tracker=tracker, endpoint='https://api.example.org/v1',
            permission=check_llm_use(access='open_access', license_type='cc-by',
                                     processing='api'))
        assert tracker.to_dict()['permissions']['unnamed']['endpoint'] == 'https://api.example.org/v1'


class TestCredentialsNotRecorded:
    def test_userinfo_and_query_are_removed(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        run(tracker=tracker, endpoint='https://key:secret@api.example.org/v1?token=abc',
            permission=check_llm_use(access='open_access', license_type='cc-by',
                                     processing='api'))
        logged = tracker.to_dict()['permissions']['unnamed']['endpoint']
        assert 'secret' not in logged and 'token' not in logged
        assert logged == 'https://api.example.org/v1'

    def test_error_message_hides_credentials(self):
        permission = check_llm_use(access='subscription', ai_terms='silent', processing='local')
        with pytest.raises(PermissionError) as error:
            run(permission=permission, endpoint='https://key:secret@api.example.org/v1')
        assert 'secret' not in str(error.value)


class TestStrictFlags:
    @pytest.mark.parametrize('value', ['False', 'no', 0, None, float('nan')])
    def test_research_org_must_be_boolean(self, value):
        with pytest.raises(ValueError, match='research_org'):
            check_llm_use(region='eu', research_org=value)

    def test_unhashable_choice_is_a_value_error(self):
        with pytest.raises(ValueError, match='processing'):
            check_llm_use(processing=['api'])


class TestEncoderOutput:
    def test_row_count_mismatch_returns_empty(self):
        class Short:
            def encode(self, texts):
                return np.ones((1, 2))

        assert similarity_scores(['a', 'b', 'c'], 'd', Short()) == []

    def test_non_finite_output_returns_empty(self):
        class Broken:
            def encode(self, texts):
                return np.full((len(texts), 2), np.nan)

        assert similarity_scores(['a', 'b'], 'd', Broken()) == []


class TestPageText:
    def test_page_without_text_is_skipped(self):
        chunks = chunk_pages([{'page': 1}, {'page': 2, 'text': 'second page'}])
        assert [c['pdf_page'] for c in chunks] == [2]
