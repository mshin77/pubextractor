import hashlib

import numpy as np

from pubextractor.licensing import check_llm_use
from pubextractor.provenance import ProvenanceTracker, log_human
from pubextractor.rag import extract_with_rag


class Encoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 2))


def responder(model):
    def respond(prompt, images=None):
        return 'not found'
    respond.model = model
    return respond


class TestPromptRecords:
    def test_every_prompt_is_hashed_with_its_model(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        tracker.log_prompt('first', model='m1')
        tracker.log_prompt('second', model='m2')
        records = tracker.to_dict()['prompts']['records']
        assert [r['model'] for r in records] == ['m1', 'm2']
        assert records[1]['sha256'] == hashlib.sha256(b'second').hexdigest()
        assert 'second' not in str(records)

    def test_rag_records_model_name(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        extract_with_rag([{'page': 1, 'text': 'some text here'}], {'a': 'q', 'b': 'r'},
                         responder('gemma3:4b'), Encoder(), tracker=tracker,
                         permission='unchecked')
        assert [r['model'] for r in tracker.to_dict()['prompts']['records']] == ['gemma3:4b'] * 2


class TestPermissionsPerPaper:
    def test_each_paper_keeps_its_verdict(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        pages = [{'page': 1, 'text': 'some text here'}]
        local = check_llm_use(access='subscription', ai_terms='silent')
        open_api = check_llm_use(access='open_access', license_type='cc-by', processing='api')
        extract_with_rag(pages, {'a': 'q'}, responder('m'), Encoder(), tracker=tracker,
                         permission=local, endpoint='http://localhost:11434/v1', paper_id='S01')
        extract_with_rag(pages, {'a': 'q'}, responder('m'), Encoder(), tracker=tracker,
                         permission=open_api, endpoint='https://api.openai.com/v1',
                         paper_id='S02')
        permissions = tracker.to_dict()['permissions']
        assert permissions['S01']['verdict'] == 'local_only'
        assert permissions['S02']['verdict'] == 'any_llm'
        assert 'permission' not in tracker.to_dict()


class TestHumanSteps:
    def test_corrections_are_logged(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        log_human(tracker, 'S07', 'sd_post_control', 'correct', 12.4, 14.2, coder='MS')
        step = tracker.to_dict()['human'][0]
        assert step['field'] == 'sd_post_control' and step['after'] == 14.2
        assert step['coder'] == 'MS' and step['paper_id'] == 'S07'
        assert step['time']
