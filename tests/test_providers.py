import base64
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from pubextractor.providers import PROVIDERS, load_env, make_responder


@pytest.fixture
def isolated_git(monkeypatch):
    if shutil.which('git') is None:
        pytest.skip('git is not installed')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')


class TestLoadEnv:
    def test_sets_missing_variables_and_returns_names(self, tmp_path):
        env_file = tmp_path / '.env'
        env_file.write_text('# keys\nGEMINI_API_KEY="abc"\nexport OPENAI_API_KEY=xyz\n\nBAD LINE\n')
        environ = {}
        names = load_env(env_file, environ)
        assert names == ['GEMINI_API_KEY', 'OPENAI_API_KEY']
        assert environ == {'GEMINI_API_KEY': 'abc', 'OPENAI_API_KEY': 'xyz'}

    def test_existing_variables_win(self, tmp_path):
        env_file = tmp_path / '.env'
        env_file.write_text('OPENAI_API_KEY=from_file\n')
        environ = {'OPENAI_API_KEY': 'already_set'}
        assert load_env(env_file, environ) == []
        assert environ['OPENAI_API_KEY'] == 'already_set'

    def test_bom_keeps_first_key_name(self, tmp_path):
        env_file = tmp_path / '.env'
        env_file.write_bytes(b'\xef\xbb\xbfOPENAI_API_KEY=abc\n')
        environ = {}
        assert load_env(env_file, environ) == ['OPENAI_API_KEY']
        assert environ == {'OPENAI_API_KEY': 'abc'}

    def test_utf16_file_is_refused_without_values(self, tmp_path):
        env_file = tmp_path / '.env'
        env_file.write_text('OPENAI_API_KEY=secretvalue\n', encoding='utf-16')
        with pytest.raises(ValueError, match='UTF-8') as caught:
            load_env(env_file, {})
        assert 'secretvalue' not in str(caught.value)

    def test_missing_file_sets_nothing(self, tmp_path):
        assert load_env(tmp_path / '.env', {}) == []

    def test_warns_when_git_does_not_ignore_the_file(self, tmp_path, isolated_git):
        subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
        env_file = tmp_path / '.env'
        env_file.write_text('OPENAI_API_KEY=x\n')
        with pytest.warns(UserWarning, match='gitignore'):
            load_env(env_file, {})

    def test_no_warning_when_ignored(self, tmp_path, recwarn, isolated_git):
        subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
        (tmp_path / '.gitignore').write_text('.env\n')
        env_file = tmp_path / '.env'
        env_file.write_text('OPENAI_API_KEY=x\n')
        load_env(env_file, {})
        assert not [w for w in recwarn if 'gitignore' in str(w.message)]


def fake_post(calls, reply='answer'):
    def post(url, body, headers, timeout):
        calls.append({'url': url, 'body': body, 'headers': headers})
        return {'choices': [{'message': {'content': reply}}]}
    return post


def fake_post_raw(calls, reply):
    def post(url, body, headers, timeout):
        calls.append({'url': url, 'body': body, 'headers': headers})
        return reply
    return post


class TestMakeResponder:
    def test_api_provider_reads_key_and_returns_endpoint(self):
        calls = []
        respond, endpoint = make_responder('gemini', 'gemini-2.5-flash',
                                           environ={'GEMINI_API_KEY': 'k1'}, post=fake_post(calls))
        assert endpoint == PROVIDERS['gemini'][0]
        assert respond('What design?') == 'answer'
        assert calls[0]['url'] == endpoint + '/chat/completions'
        assert calls[0]['headers']['Authorization'] == 'Bearer k1'
        assert calls[0]['body'] == {'model': 'gemini-2.5-flash',
                                    'messages': [{'role': 'user', 'content': 'What design?'}]}
        assert (respond.model, respond.provider) == ('gemini-2.5-flash', 'gemini')

    def test_missing_key_names_the_variable_only(self):
        with pytest.raises(ValueError, match='OPENAI_API_KEY'):
            make_responder('openai', 'gpt-4.1', environ={})

    def test_ollama_needs_no_key_and_is_local(self):
        from pubextractor.licensing import is_local_endpoint

        calls = []
        respond, endpoint = make_responder('ollama', 'gemma3:4b', environ={},
                                           post=fake_post(calls))
        respond('q')
        assert is_local_endpoint(endpoint)
        assert 'Authorization' not in calls[0]['headers']

    def test_custom_base_url(self):
        calls = []
        respond, endpoint = make_responder('custom', 'm', base_url='http://127.0.0.1:8000/v1/',
                                           environ={}, post=fake_post(calls))
        respond('q')
        assert calls[0]['url'] == 'http://127.0.0.1:8000/v1/chat/completions'

    def test_unknown_provider_without_base_url(self):
        with pytest.raises(ValueError, match='base_url'):
            make_responder('nowhere', 'm', environ={})

    def test_images_are_sent_as_base64_parts(self):
        calls = []
        respond, _ = make_responder('ollama', 'gemma3:4b', environ={}, post=fake_post(calls))
        respond('Read the axis', images=[b'\x89PNG fake'])
        content = calls[0]['body']['messages'][0]['content']
        assert content[0] == {'type': 'text', 'text': 'Read the axis'}
        url = content[1]['image_url']['url']
        assert url.startswith('data:image/png;base64,')
        assert base64.b64decode(url.split(',', 1)[1]) == b'\x89PNG fake'

    def test_http_error_hides_key_and_query(self, monkeypatch):
        from urllib import error

        from pubextractor import providers

        def raising(*args, **kwargs):
            raise error.HTTPError('https://api.openai.com/v1/chat/completions?key=s3cret',
                                  401, 'Unauthorized', {}, None)

        monkeypatch.setattr(providers._OPENER, 'open', raising)
        respond, _ = make_responder('openai', 'gpt-4.1', environ={'OPENAI_API_KEY': 's3cret'})
        with pytest.raises(RuntimeError) as caught:
            respond('q')
        assert '401' in str(caught.value)
        assert 's3cret' not in str(caught.value)

    def test_timeout_raises_runtime_error_with_redacted_url(self, monkeypatch):
        from pubextractor import providers

        def raising(*args, **kwargs):
            raise TimeoutError('timed out')

        monkeypatch.setattr(providers._OPENER, 'open', raising)
        respond, _ = make_responder('openai', 'gpt-4.1', environ={'OPENAI_API_KEY': 'secret_key'})
        with pytest.raises(RuntimeError) as caught:
            respond('q')
        error_msg = str(caught.value)
        assert 'unreachable' in error_msg
        assert 'api.openai.com' in error_msg
        assert 'secret_key' not in error_msg

    def test_malformed_json_reply_raises_runtime_error(self, monkeypatch):
        from pubextractor import providers

        def raising_json(*args, **kwargs):
            raise json.JSONDecodeError('msg', 'doc', 0)

        monkeypatch.setattr(providers._OPENER, 'open', raising_json)
        respond, _ = make_responder('ollama', 'm', environ={})
        with pytest.raises(RuntimeError) as caught:
            respond('q')
        assert 'gave no reply text' in str(caught.value)

    def test_missing_choices_in_reply_raises_runtime_error(self):
        calls = []
        respond, endpoint = make_responder('ollama', 'm', environ={},
                                           post=fake_post_raw(calls, reply={}))
        with pytest.raises(RuntimeError) as caught:
            respond('q')
        assert 'gave no reply text' in str(caught.value)

    def test_empty_choices_list_raises_runtime_error(self):
        calls = []
        respond, endpoint = make_responder('ollama', 'm', environ={},
                                           post=fake_post_raw(calls, reply={'choices': []}))
        with pytest.raises(RuntimeError) as caught:
            respond('q')
        assert 'gave no reply text' in str(caught.value)

    def test_none_content_in_reply_raises_runtime_error(self):
        calls = []
        respond, endpoint = make_responder('ollama', 'm', environ={},
                                           post=fake_post_raw(calls, reply={'choices': [{'message': {'content': None}}]}))
        with pytest.raises(RuntimeError) as caught:
            respond('q')
        assert 'gave no reply text' in str(caught.value)

    def test_base_url_without_key_env_for_keyed_provider_raises(self):
        with pytest.raises(ValueError, match='key_env'):
            make_responder('openai', 'gpt-4.1', base_url='http://127.0.0.1:8000/v1',
                           environ={'OPENAI_API_KEY': 'secret_key'}, post=fake_post([]))

    def test_base_url_with_explicit_key_env_sends_key(self):
        calls = []
        respond, endpoint = make_responder('openai', 'gpt-4.1',
                                           base_url='http://127.0.0.1:8000/v1',
                                           key_env='OPENAI_API_KEY',
                                           environ={'OPENAI_API_KEY': 'secret_key'},
                                           post=fake_post(calls))
        respond('q')
        assert calls[0]['headers']['Authorization'] == 'Bearer secret_key'


def test_package_exports_new_names():
    import pubextractor

    for name in ('load_env', 'make_responder', 'PROVIDERS', 'read_docx',
                 'read_docx_images', 'read_pdf_tables'):
        assert name in pubextractor.__all__
        assert hasattr(pubextractor, name)


class StubEncoder:
    def encode(self, texts, **kwargs):
        return np.array([[len(t) % 7 + 1.0, len(t) % 3 + 1.0] for t in texts])


def test_local_model_end_to_end_records_permission_and_model():
    from pubextractor.licensing import check_llm_use
    from pubextractor.provenance import ProvenanceTracker
    from pubextractor.rag import extract_with_rag

    calls = []

    def fake(url, body, headers, timeout):
        calls.append(url)
        return {'choices': [{'message': {'content': 'ok'}}]}

    respond, endpoint = make_responder('ollama', 'm', environ={}, post=fake)
    tracker = ProvenanceTracker('extraction', 'end-to-end')
    pages = [{'page': 1, 'text': 'Fifty students took part in the study.'}]
    extract_with_rag(pages, {'a': 'q'}, respond, StubEncoder(), tracker=tracker,
                     permission=check_llm_use(access='subscription', ai_terms='silent'),
                     endpoint=endpoint, paper_id='S01')
    record = tracker.to_dict()
    assert calls
    assert record['permissions']['S01']['verdict'] == 'local_only'
    assert record['prompts']['records'][0]['model'] == 'm'
